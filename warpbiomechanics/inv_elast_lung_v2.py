"""
inv_elast_lung_v2.py — Inverse FEM for lung elasticity from DIR-derived displacement

PHYSICS / PROBLEM FORMULATION
──────────────────────────────
Given:
  • u_target(x)  — displacement field from DIR, defined at every mesh node
  • ν             — Poisson's ratio (fixed, e.g. 0.43 for lung parenchyma)

Find:
  • E(x)          — per-element Young's modulus

by minimising the Neo-Hookean equilibrium-residual loss

    L(E)  =  ‖r(E)‖²,    r_i(E) = ∫_Ω P(F(u_target); E) : ∇φ_i dV

where  φ_i  are displacement test-function DOFs and

    P = μF + (λ(J−1) − μ) F^{−T}   (stable Neo-Hookean, Smith et al. 2018)

Physical interpretation:
  At a true equilibrium (div P = 0, zero body forces), r ≡ 0.
  Minimising ‖r‖² finds E that best satisfies equilibrium for the
  observed deformation — i.e., the stiffness map consistent with the lung
  having been a passive elastic body during the DIR-captured breathing cycle.

ALTERNATIVE LOSS (not implemented here — see [Blocker B]):
  ‖u_sim − u_target‖² requires a nested 3D forward solve (Newton iterations)
  with surface Dirichlet BCs from DIR.  See NOTES.md §Blocker B for the plan.

SECTIONS
─────────
  1. Config / argument parsing
  2. I/O   — load NIfTI images, resample isotropically
  3. Mesh  — cudaSetConnect → warp.fem.Tetmesh
  4. FEM   — function spaces, displacement and E fields
  5. Loss  — equilibrium residual integrand + squared-norm kernel
  6. Optim — SGD/Adam optimisation loop, E positivity enforcement
  7. Out   — E.npy, E_volume.nii.gz, fem_mesh.usd

KNOWN BLOCKERS / CAVEATS
─────────────────────────
[A] DVF axis convention: [dw, du, dv] → mesh (x,y,z) was empirically
    validated for the 10123_002BERFR dataset.  Verify for new subjects
    by checking that deformed surface positions match anatomy.

[B] ‖u_sim − u_target‖² loss: needs a nested 3D NH forward solve.
    Pattern: warp/examples/fem/example_elastic_shape_optimization.py
    adapted to 3D Tetmesh with surface Dirichlet BCs from DIR.

[C] meshifyVolume() is O(Nx·Ny·Nz) in pure Python.  Acceptable at
    sampling ≥ 4 mm; consider TetGen / fTetWild for finer grids.

[D] Inverted elements (J ≤ 0): wp.inverse(F) is undefined.  Check
    connector.tetras quality or add a Tikhonov regulariser F += ε·I.

[E] E parameterisation: currently clamped post-step to [E_min, E_max].
    For large-scale problems prefer log-reparameterisation (E = E_ref·exp(θ))
    so positivity is automatic and Adam momentum is preserved across clamps.
"""

import argparse
import os
import sys

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import map_coordinates

import warp as wp
import warp.fem as fem
import warp.optim

# Allow importing cudaSetConnect.py and py_cuda_helper.py from the parent dir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cudaSetConnect  # noqa: E402 (after sys.path manipulation)

from warp.render.render_usd import UsdRenderer  # noqa: E402

wp.init()


# ════════════════════════════════════════════════════════════════════════════════
# 1  CONFIG
# ════════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inverse FEM: lung elasticity from DIR displacement",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- I/O ---
    p.add_argument("--mask", required=True, help="Lung mask NIfTI (.nii.gz)")
    p.add_argument("--u",    required=True, help="DVF U (x/LR)  component NIfTI")
    p.add_argument("--v",    required=True, help="DVF V (y/AP)  component NIfTI")
    p.add_argument("--w",    required=True, help="DVF W (z/SI)  component NIfTI")
    # --- HU-based E initialisation (optional) ---
    p.add_argument("--rv_img",   default=None,
                   help="RV CT NIfTI for HU-based E initialisation (optional)")
    p.add_argument("--hu_ref_air",    type=float, default=-1000.0,
                   help="HU value representing air (maps to E_init_min)")
    p.add_argument("--hu_ref_tissue", type=float, default=0.0,
                   help="HU value representing solid tissue (maps to E_init_max)")
    p.add_argument("--E_init_min",    type=float, default=0.5,
                   help="Minimum E from HU mapping (before global [E_min,E_max] clamp)")
    p.add_argument("--E_init_max",    type=float, default=20.0,
                   help="Maximum E from HU mapping (before global [E_min,E_max] clamp)")
    p.add_argument("--E_from_hu_mode", default="clipped_linear",
                   choices=["clipped_linear"],
                   help="HU→E conversion mode")
    p.add_argument("--save_initial_E_nifti", action="store_true",
                   help="Save HU-derived initial E field as E_init_volume.nii.gz")
    # --- pre-processing ---
    p.add_argument("--sampling", type=float, default=8.0,
                   help="Isotropic resampling resolution (mm)")
    # --- material ---
    p.add_argument("--nu",    type=float, default=0.43,
                   help="Poisson ratio (fixed for all elements)")
    p.add_argument("--E_init", type=float, default=1.0,
                   help="Fallback uniform E_init when --rv_img is not provided")
    p.add_argument("--E_min",  type=float, default=1e-3,
                   help="Young's modulus global lower clamp (optimisation + init)")
    p.add_argument("--E_max",  type=float, default=1e3,
                   help="Young's modulus global upper clamp (optimisation + init)")
    # --- optimisation ---
    p.add_argument("--lr",       type=float, default=1e-3,
                   help="Optimiser learning rate")
    p.add_argument("--momentum", type=float, default=0.0,
                   help="SGD momentum factor (0 = vanilla SGD)")
    p.add_argument("--dampening", type=float, default=0.0,
                   help="SGD momentum dampening")
    p.add_argument("--nesterov", action="store_true",
                   help="Use Nesterov momentum (requires --momentum > 0)")
    p.add_argument("--n_iters", type=int,   default=200,
                   help="Optimisation iterations")
    # --- output ---
    p.add_argument("--out_dir", default="./outputs",
                   help="Output directory")
    p.add_argument("--usd_name", default="fem_mesh.usd",
                   help="Filename for the USD mesh render")
    # --- warp ---
    p.add_argument("--device", default=None,
                   help="Warp device override (e.g. 'cuda:0')")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════════════
# 2  I/O — load NIfTI images and resample to isotropic spacing
# ════════════════════════════════════════════════════════════════════════════════

def make_isotropic(
    image: sitk.Image,
    sampling: float = 1.0,
    interpolator=sitk.sitkLinear,
) -> sitk.Image:
    """Resample *image* to isotropic voxel spacing = *sampling* mm."""
    orig_spacing = image.GetSpacing()
    orig_size    = image.GetSize()
    new_spacing  = [sampling] * image.GetDimension()
    new_size     = [
        int(round(sz * spc / sampling + 0.3))
        for sz, spc in zip(orig_size, orig_spacing)
    ]
    return sitk.Resample(
        image, new_size,
        sitk.Transform(), interpolator,
        image.GetOrigin(), new_spacing,
        image.GetDirection(), 0,
        image.GetPixelID(),
    )


def load_inputs(args: argparse.Namespace):
    """
    Load mask + DVF from NIfTI, resample to isotropic.

    Returns
    -------
    volume    : np.ndarray uint8 [x, y, z], binary mask
    u_np      : np.ndarray float32 [x, y, z], DIR x-component
    v_np      : np.ndarray float32 [x, y, z], DIR y-component
    w_np      : np.ndarray float32 [x, y, z], DIR z-component
    mask_sitk : sitk.Image — resampled mask (used to write output NIfTI)
    """
    print(f"  Loading mask:  {args.mask}")
    print(f"  Loading U/V/W: {args.u} / {args.v} / {args.w}")

    mask_sitk = make_isotropic(
        sitk.ReadImage(args.mask), args.sampling,
        interpolator=sitk.sitkNearestNeighbor,
    )
    u_sitk = make_isotropic(sitk.ReadImage(args.u), args.sampling)
    v_sitk = make_isotropic(sitk.ReadImage(args.v), args.sampling)
    w_sitk = make_isotropic(sitk.ReadImage(args.w), args.sampling)

    # ITK/SimpleITK stores arrays as [z, y, x] in numpy; transpose to [x, y, z]
    def to_xyz(img: sitk.Image) -> np.ndarray:
        return np.transpose(sitk.GetArrayFromImage(img), (2, 1, 0))

    volume = (to_xyz(mask_sitk) > 0).astype(np.uint8)
    u_np   = to_xyz(u_sitk).astype(np.float32)
    v_np   = to_xyz(v_sitk).astype(np.float32)
    w_np   = to_xyz(w_sitk).astype(np.float32)

    print(f"  Volume shape (x,y,z): {volume.shape}, "
          f"non-zero voxels: {volume.sum()}")
    return volume, u_np, v_np, w_np, mask_sitk


# ════════════════════════════════════════════════════════════════════════════════
# 2b  HU-BASED E INITIALISATION (optional)
# ════════════════════════════════════════════════════════════════════════════════

def load_rv_ct(args: argparse.Namespace, mask_sitk: sitk.Image) -> np.ndarray:
    """
    Load RV CT image with nibabel, resample to the same isotropic grid as the
    mask with SimpleITK, and return the HU array in [x, y, z] voxel order.

    Nibabel is used for loading (array + affine + header), so all raw metadata
    is available.  SimpleITK handles resampling for coordinate-system consistency
    with the mask pipeline.

    Returns
    -------
    rv_xyz : np.ndarray float32 [x, y, z] — HU values on the mask voxel grid
    """
    print(f"  nibabel: loading {args.rv_img}")
    nib_rv  = nib.load(args.rv_img)
    zooms   = nib_rv.header.get_zooms()[:3]
    print(f"  nibabel: shape={nib_rv.shape}, zooms={tuple(f'{z:.2f}' for z in zooms)} mm")

    # Resample RV CT to same isotropic grid as mask using SimpleITK.
    # sitk.sitkLinear is correct for continuous HU values.
    rv_sitk = make_isotropic(
        sitk.ReadImage(args.rv_img), args.sampling,
        interpolator=sitk.sitkLinear,
    )
    rv_xyz = np.transpose(
        sitk.GetArrayFromImage(rv_sitk), (2, 1, 0)
    ).astype(np.float32)   # [z,y,x] → [x,y,z]

    print(f"  RV CT (x,y,z): {rv_xyz.shape},  "
          f"HU range [{rv_xyz.min():.0f}, {rv_xyz.max():.0f}]")
    return rv_xyz


def hu_to_E(
    hu:             np.ndarray,
    hu_ref_air:     float,
    hu_ref_tissue:  float,
    E_init_min:     float,
    E_init_max:     float,
    mode:           str = "clipped_linear",
) -> np.ndarray:
    """
    Convert HU values to Young's modulus estimates.

    mode = "clipped_linear"
    ───────────────────────
    Maps [hu_ref_air, hu_ref_tissue] linearly to [E_init_min, E_init_max],
    clipped outside that range.

        α = clip((HU − HU_air) / (HU_tissue − HU_air), 0, 1)
        E = E_init_min + (E_init_max − E_init_min) * α

    Physical rationale for lung CT (typical values):
      • HU ≈ −950 to −800  → aerated alveoli (low density, low E)
      • HU ≈ −200 to  +50  → atelectasis / infiltrate (high density, high E)

    Returns
    -------
    E_init : np.ndarray float32, same shape as *hu*
    """
    if mode == "clipped_linear":
        span = hu_ref_tissue - hu_ref_air          # usually 1000 HU
        alpha = np.clip((hu - hu_ref_air) / span, 0.0, 1.0)
        E_init = E_init_min + (E_init_max - E_init_min) * alpha
    else:
        raise ValueError(f"Unknown E_from_hu_mode: {mode!r}")
    return E_init.astype(np.float32)


def init_E_from_hu(
    connector,
    rv_xyz:     np.ndarray,
    sampling:   float,
    args:       argparse.Namespace,
) -> np.ndarray:
    """
    Build a per-element initial Young's modulus array from CT HU values.

    Algorithm
    ---------
    1. For each mesh node, find its voxel index: ix = round(pos_x / sampling).
    2. Sample HU from *rv_xyz* at that voxel (nearest-neighbour; the positions
       are already on the mesh grid so sub-voxel error is negligible).
    3. For each tet, average the HU of its 4 vertices → mean_HU_tet.
    4. Convert mean_HU_tet → E via hu_to_E().
    5. Clamp result to [args.E_min, args.E_max].

    GPU note: steps 1-4 are performed on CPU as numpy operations.  For meshes
    with > ~1 M elements this loop could be vectorised with np.take or moved to
    a wp.kernel; at typical lung FEM resolutions (≤ 200 k tets) it is fast enough.

    Returns
    -------
    E_init_array : np.ndarray float32, shape [n_tets]
    """
    positions_np = connector.getVolumePoints()   # [n_nodes, 3] mm
    nx, ny, nz   = rv_xyz.shape

    # Voxel indices for every node (nearest-neighbour on the isotropic grid)
    ix = np.clip(np.round(positions_np[:, 0] / sampling).astype(int), 0, nx - 1)
    iy = np.clip(np.round(positions_np[:, 1] / sampling).astype(int), 0, ny - 1)
    iz = np.clip(np.round(positions_np[:, 2] / sampling).astype(int), 0, nz - 1)

    node_hu = rv_xyz[ix, iy, iz]   # [n_nodes]

    # Per-tet: mean HU of the 4 vertices
    # connector.tetras shape: [n_tets, 4], node indices already mapped
    tet_np     = connector.tetras              # [n_tets, 4]
    tet_hu     = node_hu[tet_np]              # [n_tets, 4]
    mean_hu    = tet_hu.mean(axis=1)          # [n_tets]

    E_init_arr = hu_to_E(
        mean_hu,
        hu_ref_air    = args.hu_ref_air,
        hu_ref_tissue = args.hu_ref_tissue,
        E_init_min    = args.E_init_min,
        E_init_max    = args.E_init_max,
        mode          = args.E_from_hu_mode,
    )
    # Global clamp so the initial values also respect [E_min, E_max]
    E_init_arr = np.clip(E_init_arr, args.E_min, args.E_max)

    print(f"  HU-based E init: min={E_init_arr.min():.4f}  "
          f"max={E_init_arr.max():.4f}  mean={E_init_arr.mean():.4f}")
    return E_init_arr


# ════════════════════════════════════════════════════════════════════════════════
# 3  MESH — binary mask → tetrahedral mesh via cudaSetConnect
# ════════════════════════════════════════════════════════════════════════════════

def build_mesh(
    volume: np.ndarray,
    sampling: float,
):
    """
    Build a tetrahedral mesh from a binary lung mask.

    Uses cudaSetConnect, which performs:
      1. findConnections()  — GPU spring connectivity (needed for surface detection)
      2. meshifyVolume()    — Marching-tets on the binary grid

    [Blocker C] meshifyVolume() is a Python loop over every voxel — O(Nx·Ny·Nz).
    For large volumes at fine spacing, consider a compiled mesher (TetGen, fTetWild).

    Returns
    -------
    connector : SetConnect — holds .nzindices for DVF node mapping
    geo       : warp.fem.Tetmesh
    positions : wp.array(dtype=wp.vec3)
    tets      : wp.array(dtype=int32), shape [n_tets * 4]
    """
    # cudaSetConnect expects a Fortran-order (column-major) flattened array
    connector = cudaSetConnect.SetConnect(
        arraySize=volume.shape,
        iso_resolution=sampling,
        volume=volume.flatten(order="F"),
    )
    connector.findConnections()
    connector.meshifyVolume()

    positions_np = connector.getVolumePoints()          # [n_nodes, 3]  float32
    tets_np      = connector.tetras.astype(np.int32)    # [n_tets, 4]   int32

    positions = wp.array(positions_np, dtype=wp.vec3)
    tets      = wp.array(tets_np)                       # shape [n_tets*4] for Tetmesh

    geo = fem.Tetmesh(tet_vertex_indices=tets, positions=positions)

    print(f"  Nodes: {positions.shape[0]},  Tets: {tets_np.shape[0]}")
    return connector, geo, positions, tets


def map_dvf_to_mesh(
    connector,
    u_np: np.ndarray,
    v_np: np.ndarray,
    w_np: np.ndarray,
) -> wp.array:
    """
    Sample the DIR displacement field at mesh node positions.

    Axis convention [Blocker A]:
      cudaSetConnect places particles in Fortran-order (x fastest).
      After empirical validation on the 10123_002BERFR dataset the mapping is:
        mesh x  ←  w  (z / SI component of DIR)
        mesh y  ←  u  (x / LR component of DIR)
        mesh z  ←  v  (y / AP component of DIR)
      Verify this for new subjects by overlaying the deformed mesh on anatomy.

    Returns
    -------
    dvf : wp.array(dtype=wp.vec3), shape [n_nodes]
    """
    # .flatten(order='F') matches the Fortran-order indexing used in cudaSetConnect
    du = u_np.flatten(order="F")[connector.nzindices]
    dv = v_np.flatten(order="F")[connector.nzindices]
    dw = w_np.flatten(order="F")[connector.nzindices]

    # [Blocker A] axis permutation — see docstring above
    dvf_np = np.stack([dw, du, dv], axis=1).astype(np.float32)   # [n_nodes, 3]
    dvf    = wp.array(dvf_np, dtype=wp.vec3)

    assert dvf.shape[0] == connector.numParticles, (
        f"DVF / mesh node count mismatch: {dvf.shape[0]} vs {connector.numParticles}"
    )
    return dvf


# ════════════════════════════════════════════════════════════════════════════════
# 4  FEM SETUP — function spaces and field initialisation
# ════════════════════════════════════════════════════════════════════════════════

def build_fem_fields(
    geo,
    dvf:            wp.array,
    E_init:         float,
    E_init_array:   np.ndarray = None,
):
    """
    Construct FEM function spaces and initialise fields.

    Spaces
    ------
    u_space : degree-1 continuous Lagrange on tets, dtype=wp.vec3
              (displacement; one vec3 DOF per vertex)
    E_space : degree-0 discontinuous (piecewise-constant), dtype=float
              (Young's modulus; one float DOF per element)

    The displacement field u_field is SET to u_target from DIR and is NOT
    optimised.  E_field is the optimisation variable.

    Parameters
    ----------
    E_init_array : np.ndarray float32 [n_tets], optional
        Per-element initial Young's modulus from HU mapping.
        When None, E_field is initialised uniformly to *E_init*.

    Returns
    -------
    u_field : DiscreteField — fixed to observed displacement
    E_field : DiscreteField — initial E, requires_grad=True
    v_test  : TestField  — test function for the residual linear form
    """
    u_space = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec3)
    E_space = fem.make_polynomial_space(
        geo, degree=0, dtype=float, discontinuous=True
    )

    u_field = u_space.make_field()
    E_field = E_space.make_field()

    # Fix displacement to observed DIR field (not optimised)
    u_field.dof_values.assign(dvf)

    # Initialise E — per-element from HU if available, otherwise uniform
    if E_init_array is not None:
        E_field.dof_values.assign(wp.array(E_init_array, dtype=float))
        print(f"  E initialised from HU map ({E_init_array.shape[0]} elements)")
    else:
        E_field.dof_values.fill_(E_init)
        print(f"  E initialised uniformly to {E_init}")
    E_field.dof_values.requires_grad = True

    # Test function for assembling the internal-force residual vector
    v_test = fem.make_test(space=u_space)

    print(f"  u DOFs: {u_space.node_count()},  E DOFs: {E_space.node_count()}")
    return u_field, E_field, v_test


# ════════════════════════════════════════════════════════════════════════════════
# 5  LOSS — Neo-Hookean equilibrium residual  ‖r(E)‖²
# ════════════════════════════════════════════════════════════════════════════════

@fem.integrand
def pk1_residual_form(
    s: fem.Sample,
    u: fem.Field,
    E: fem.Field,
    v: fem.Field,
    nu: float,
):
    """
    FEM internal-force linear form:
        r_i = ∫_Ω P(F(u); E) : ∇φ_i dV

    where P is the first Piola-Kirchhoff stress for stable Neo-Hookean:
        P = μ F + (λ(J−1) − μ) F^{−T}

    [Blocker D] If any tet has J ≤ 0 (inverted element), wp.inverse(F)
    is undefined and will produce NaN.  Consider pre-filtering tets
    in connector.meshifyVolume() or adding ε·I to F before inversion.
    """
    F   = wp.identity(n=3, dtype=float) + fem.grad(u, s)
    J   = wp.determinant(F)

    E_val = E(s)
    mu    = E_val / (2.0 * (1.0 + nu))
    lam   = E_val * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # First Piola-Kirchhoff stress (stable Neo-Hookean)
    F_inv_T = wp.transpose(wp.inverse(F))
    P       = mu * F + (lam * (J - 1.0) - mu) * F_inv_T

    # Scalar contribution to DOF i: P : ∇φ_i
    return wp.ddot(P, fem.grad(v, s))


@wp.kernel
def squared_norm_vec3(
    r:    wp.array(dtype=wp.vec3),
    loss: wp.array(dtype=float),
):
    """
    Accumulate Σ ‖r_i‖² into loss[0] (differentiable through wp.atomic_add).
    Backward: r_i.grad += 2 * r_i * loss.grad[0]
    """
    i = wp.tid()
    # wp.atomic_add(loss, 0, wp.squared_norm(r[i]))
    wp.atomic_add(loss, 0, wp.length_sq(r[i]))


# ─── regularisation hook ────────────────────────────────────────────────────
# Uncomment and connect to the optimisation loop to add a smoothness prior.
#
# @fem.integrand
# def tikhonov_reg(s: fem.Sample, E: fem.Field, E_ref: float, weight: float):
#     """Penalise deviation from a reference E to prevent wild spatial variation."""
#     delta = E(s) - E_ref
#     return weight * delta * delta
#
# Usage inside optimisation loop (before tape.backward):
#   fem.integrate(tikhonov_reg,
#                 domain=fem.Cells(geo),
#                 fields={"E": E_field},
#                 values={"E_ref": args.E_init, "weight": 1e-4},
#                 output=loss, add=True)
# ────────────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════════
# 6  OPTIMISATION
# ════════════════════════════════════════════════════════════════════════════════

def run_optimization(
    geo,
    u_field,
    E_field,
    v_test,
    nu:        float,
    E_min:     float,
    E_max:     float,
    lr:        float,
    n_iters:   int,
    momentum:  float = 0.0,
    dampening: float = 0.0,
    nesterov:  bool  = False,
):
    """
    Gradient-based inversion for per-element Young's modulus.

    Loss:   L(E) = ‖r(E)‖²   (equilibrium residual squared norm)
    Optim:  SGD  (with optional momentum / Nesterov)

    SGD vs Adam for inverse FEM elasticity
    ───────────────────────────────────────
    Why Adam was chosen originally:
      • Adaptive per-parameter learning rates handle the large E variation
        across tissue types (kPa–MPa range) without hand-tuning lr.
      • Fast convergence on ill-conditioned problems (heterogeneous curvature).

    Advantages of SGD here:
      • Simpler, more predictable update step: Δθ = −lr · g  (+ momentum term).
      • Smoother E fields — Adam's per-parameter scaling can amplify noisy
        gradients in low-residual elements, leading to speckle artefacts.
      • Easier to interpret physically: the update is proportional to the
        equilibrium-residual gradient, so the stiffness evolves "naturally".
      • Momentum (β ≈ 0.9) accelerates convergence in well-conditioned regions
        while remaining stable, unlike Adam's sqrt(v) denominator which can
        overshoot when gradients change sign rapidly.

    Disadvantages of SGD:
      • Requires careful lr tuning — too large → divergence; too small → slow.
      • Without momentum, convergence near the minimum is slow (linear rate).

    Recommended lr ranges for this equilibrium-residual setting:
      • No momentum   (momentum=0):     lr ≈ 1e-4 to 5e-3
      • With momentum (momentum≈0.9):   lr ≈ 1e-4 to 1e-3
      • Start with lr=1e-3; if loss oscillates, halve it.
      • If E collapses to E_min after a few iterations, lr is too large.

    E is clamped to [E_min, E_max] after each step via in-place assign().
    This preserves the array object so SGD's momentum buffer remains valid.

    [Blocker B] ‖u_sim − u_target‖² alternative:
      Replace this block with a nested forward solve using the pattern from
      warp/examples/fem/example_elastic_shape_optimization.py:
        1. Identify surface nodes (fem.BoundarySides + classify by norm direction).
        2. Build boundary projector for surface Dirichlet BCs from DIR.
        3. Assemble 3D NH stiffness matrix K(E).
        4. Record tape.record_func() for the linear solve adjoint.
        5. Compute interior residual: ‖u_interior_sim − u_interior_target‖²
        6. Backward + SGD step on E.
    """
    n_nodes = u_field.dof_values.shape[0]

    # SGD: initialised with E_field.dof_values; we use .assign() throughout
    # so the array object (and its momentum buffer) remains the same.
    # warp.optim.SGD(params, lr, momentum, dampening, weight_decay, nesterov)
    opt = warp.optim.SGD(
        [E_field.dof_values],
        lr        = lr,
        momentum  = momentum,
        dampening = dampening,
        nesterov  = nesterov,
    )

    optim_desc = f"SGD lr={lr}"
    if momentum > 0.0:
        optim_desc += f", momentum={momentum}"
        if nesterov:
            optim_desc += " (Nesterov)"
    print(f"  Running {n_iters} iterations  [{optim_desc}]")

    for it in range(n_iters):
        # Allocate fresh residual and loss buffers each iteration.
        # r must start at zero because fem.integrate uses atomic adds.
        r    = wp.zeros(n_nodes, dtype=wp.vec3, requires_grad=True)
        loss = wp.zeros(1,       dtype=float,   requires_grad=True)

        tape = wp.Tape()
        with tape:
            # Assemble internal-force residual r_i = ∫ P(F;E) : ∇φ_i dV
            fem.integrate(
                pk1_residual_form,
                domain=fem.Cells(geo),
                fields={"u": u_field, "E": E_field, "v": v_test},
                values={"nu": nu},
                output=r,
            )
            # Compute loss = ‖r‖²  (also on tape for backward pass)
            wp.launch(squared_norm_vec3, dim=n_nodes, inputs=[r, loss])

        tape.backward(loss)

        # SGD step using the gradient stored in .grad by the backward pass
        opt.step([E_field.dof_values.grad])

        # Enforce positivity via in-place clamp (preserves array object for SGD)
        E_np = E_field.dof_values.numpy()
        E_np = np.clip(E_np, E_min, E_max)
        E_field.dof_values.assign(wp.array(E_np, dtype=float))
        E_field.dof_values.requires_grad = True   # re-arm after assign

        tape.zero()   # zero gradient buffers for next iteration

        if it % 10 == 0 or it == n_iters - 1:
            loss_val = float(loss.numpy()[0])
            print(
                f"  iter {it:4d}  |  loss = {loss_val:.6e}  |  "
                f"E min={E_np.min():.4f}  max={E_np.max():.4f}  "
                f"mean={E_np.mean():.4f}"
            )

    return E_field


# ════════════════════════════════════════════════════════════════════════════════
# 7  OUTPUT — save E field, mesh, and back-projected volume
# ════════════════════════════════════════════════════════════════════════════════

def save_outputs(
    geo,
    E_field,
    connector,
    mask_sitk:   sitk.Image,
    args:        argparse.Namespace,
    E_init_array: np.ndarray = None,
) -> None:
    """
    Write all outputs to args.out_dir:

    E_field.npy          — per-element E values after optimisation, shape [n_tets]
    E_volume.nii.gz      — E back-projected onto the image voxel grid
    <usd_name>           — triangulated surface mesh for USD viewer
    E_init_volume.nii.gz — HU-derived initial E (only if --save_initial_E_nifti)
    """
    os.makedirs(args.out_dir, exist_ok=True)
    E_np = E_field.dof_values.numpy()

    # -- E values per element -------------------------------------------------
    e_path = os.path.join(args.out_dir, "E_field.npy")
    np.save(e_path, E_np)
    print(f"  Saved E_field.npy  ({E_np.shape[0]} elements) → {e_path}")

    # -- USD surface mesh render -----------------------------------------------
    usd_path = os.path.join(args.out_dir, args.usd_name)
    renderer  = UsdRenderer(usd_path)
    pos_np    = geo.positions.numpy()
    tet_np    = geo.tet_vertex_indices.numpy().reshape(-1, 4)
    # Each tet contributes 4 triangular faces to the surface render
    faces = []
    for i, j, k, l in tet_np:
        faces += [[i, j, k], [i, j, l], [i, k, l], [j, k, l]]
    renderer.render_mesh(name="fem_surface", points=pos_np, indices=faces)
    renderer.save()
    print(f"  Saved mesh USD     → {usd_path}")

    # -- Back-project E to image voxel space ----------------------------------
    # Strategy: for each tet, find its centroid voxel and accumulate E.
    # Multiple tets can share a voxel; we average their E values.
    sampling  = args.sampling
    all_nodes = connector.getVolumePoints()   # [n_nodes, 3] in mm
    tet_raw   = connector.tetras              # [n_tets, 4] node indices

    E_vol   = np.zeros(connector.arraySize, dtype=np.float32)
    E_count = np.zeros(connector.arraySize, dtype=np.float32)

    for tet_idx, tet in enumerate(tet_raw):
        centroid = all_nodes[tet].mean(axis=0)   # [3] mm
        ix = int(round(centroid[0] / sampling))
        iy = int(round(centroid[1] / sampling))
        iz = int(round(centroid[2] / sampling))
        # Clamp to grid bounds
        ix = np.clip(ix, 0, connector.arraySize[0] - 1)
        iy = np.clip(iy, 0, connector.arraySize[1] - 1)
        iz = np.clip(iz, 0, connector.arraySize[2] - 1)
        E_vol[ix, iy, iz]   += E_np[tet_idx]
        E_count[ix, iy, iz] += 1

    E_count[E_count == 0] = 1            # avoid division by zero
    E_vol /= E_count

    # helper: write a [x,y,z] float32 array as NIfTI using mask geometry
    def _write_nifti(vol_xyz: np.ndarray, filename: str) -> None:
        img = sitk.GetImageFromArray(np.transpose(vol_xyz, (2, 1, 0)))
        img.CopyInformation(mask_sitk)
        path = os.path.join(args.out_dir, filename)
        sitk.WriteImage(img, path)
        print(f"  Saved {filename:<28} → {path}")

    _write_nifti(E_vol, "E_volume.nii.gz")

    # -- Optional: save HU-derived initial E as NIfTI -------------------------
    if args.save_initial_E_nifti and E_init_array is not None:
        # Back-project E_init_array to voxel space (same strategy as E_vol)
        Ei_vol   = np.zeros(connector.arraySize, dtype=np.float32)
        Ei_count = np.zeros(connector.arraySize, dtype=np.float32)
        for tet_idx, tet in enumerate(tet_raw):
            centroid = all_nodes[tet].mean(axis=0)
            ix = np.clip(int(round(centroid[0] / sampling)), 0, connector.arraySize[0]-1)
            iy = np.clip(int(round(centroid[1] / sampling)), 0, connector.arraySize[1]-1)
            iz = np.clip(int(round(centroid[2] / sampling)), 0, connector.arraySize[2]-1)
            Ei_vol[ix, iy, iz]   += E_init_array[tet_idx]
            Ei_count[ix, iy, iz] += 1
        Ei_count[Ei_count == 0] = 1
        Ei_vol /= Ei_count
        _write_nifti(Ei_vol, "E_init_volume.nii.gz")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    with wp.ScopedDevice(args.device):

        print("\n=== 2. I/O — loading inputs ===")
        volume, u_np, v_np, w_np, mask_sitk = load_inputs(args)

        # --- 2b. Optional: load RV CT and derive per-element E init from HU ---
        E_init_array = None
        if args.rv_img is not None:
            print("\n=== 2b. HU-based E initialisation ===")
            rv_xyz       = load_rv_ct(args, mask_sitk)
            # rv_xyz is loaded BEFORE mesh so it's ready once connector exists.
            # Actual per-element init happens after mesh build (connector needed).

        print("\n=== 3. Mesh — building tetrahedral mesh ===")
        connector, geo, positions, tets = build_mesh(volume, args.sampling)
        dvf = map_dvf_to_mesh(connector, u_np, v_np, w_np)

        # Compute per-element E from HU now that connector is available
        if args.rv_img is not None:
            E_init_array = init_E_from_hu(connector, rv_xyz, args.sampling, args)

        print("\n=== 4. FEM — setting up spaces and fields ===")
        u_field, E_field, v_test = build_fem_fields(
            geo, dvf, args.E_init, E_init_array=E_init_array
        )

        print("\n=== 6. Optimisation ===")
        E_field = run_optimization(
            geo, u_field, E_field, v_test,
            nu        = args.nu,
            E_min     = args.E_min,
            E_max     = args.E_max,
            lr        = args.lr,
            n_iters   = args.n_iters,
            momentum  = args.momentum,
            dampening = args.dampening,
            nesterov  = args.nesterov,
        )

        print("\n=== 7. Output ===")
        save_outputs(
            geo, E_field, connector, mask_sitk, args,
            E_init_array=E_init_array,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
