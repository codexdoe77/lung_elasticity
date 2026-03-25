# Inverse FEM — Lung Elasticity from DIR Displacement
## `inv_elast_lung_v2.py`

---

## Run instructions

### Minimal (uniform E init, vanilla SGD)
```bash
cd /cvibraid/cvib2/apps/personal/mkim3/warpbiomechanics

python test/inv_elast_lung_v2.py \
    --mask /home/marvel/Documents/winter2026/march2026/iso4/10123_002BERFR/RV_mask_iso4.0.nii.gz \
    --u    /home/marvel/Documents/winter2026/march2026/iso4/10123_002BERFR/U.nii.gz \
    --v    /home/marvel/Documents/winter2026/march2026/iso4/10123_002BERFR/V.nii.gz \
    --w    /home/marvel/Documents/winter2026/march2026/iso4/10123_002BERFR/W.nii.gz \
    --sampling 8.0 --nu 0.43 --E_init 1.0 \
    --E_min 1e-3 --E_max 1e3 \
    --lr 1e-3 --n_iters 200 \
    --out_dir ./outputs/10123_002BERFR_sgd
```

### SGD with momentum
```bash
python test/inv_elast_lung_v2.py \
    --mask ... --u ... --v ... --w ... \
    --sampling 8.0 --nu 0.43 \
    --lr 5e-4 --momentum 0.9 --n_iters 300 \
    --out_dir ./outputs/10123_002BERFR_sgd_mom09
```

### Nesterov momentum
```bash
    --lr 3e-4 --momentum 0.9 --nesterov
```

### HU-based E init (from RV CT) + SGD with momentum
```bash
python test/inv_elast_lung_v2.py \
    --mask ... --u ... --v ... --w ... \
    --rv_img /home/marvel/Documents/winter2026/march2026/iso4/10123_002BERFR/RV_img_iso4.0.nii.gz \
    --hu_ref_air -1000 --hu_ref_tissue 0 \
    --E_init_min 0.5 --E_init_max 20.0 \
    --E_from_hu_mode clipped_linear \
    --save_initial_E_nifti \
    --sampling 8.0 --nu 0.43 \
    --E_min 1e-3 --E_max 1e3 \
    --lr 5e-4 --momentum 0.9 \
    --n_iters 300 \
    --out_dir ./outputs/10123_002BERFR_hu_init
```

The script **must be launched from the `warpbiomechanics/` directory** so that
`cudaSetConnect.py` and `py_cuda_helper.py` are importable.

Outputs:
```
outputs/10123_002BERFR_8mm/
├── E_field.npy        # per-element E, shape [n_tets]
├── E_volume.nii.gz    # E back-projected to image voxel grid
└── fem_mesh.usd       # triangulated surface for USD viewer
```

---

## Physics and loss function

### Why not the energy loss from `example_inv_elast_FEM_usd.py`?

The existing script minimises `∫ Ψ(F; E) dV` w.r.t. E with u fixed.
For the Neo-Hookean potential:

    Ψ = μ ‖F‖² + λ(J−1−μ/λ)²

as E → 0, μ → 0 and λ → 0, so Ψ → 0.  The gradient ∂Ψ/∂E > 0 everywhere,
meaning SGD always drives E toward the lower clip bound.  This makes the
existing code converge to E_min regardless of the deformation field.

### What `inv_elast_lung_v2.py` implements

**Equilibrium residual loss:**

    L(E) = ‖r(E)‖²,     r_i(E) = ∫_Ω P(F(u_target); E) : ∇φ_i dV

where P is the first Piola-Kirchhoff stress:

    P = μF + (λ(J−1) − μ) F^{−T}

At true elastic equilibrium (div P = 0, zero body forces), r ≡ 0.
Minimising ‖r‖² finds E such that the observed DIR displacement field
u_target satisfies Neo-Hookean equilibrium as closely as possible.

**Gradient:**

    ∂L/∂E_j = 2 rᵀ ∂r/∂E_j

which is computed automatically by Warp's autodiff tape (chain rule through
`fem.integrate` and `wp.launch(squared_norm_vec3)`).

---

## Blockers

### [A] DVF axis convention

The existing code uses `np.stack([dw, du, dv], axis=1)` for the DVF.
This mapping (mesh-x ← w, mesh-y ← u, mesh-z ← v) was empirically validated
for the 10123_002BERFR dataset by checking that deformed node positions overlay
expected anatomy.

**How to verify for a new subject:**
1. Export deformed positions: `deformed = geo.positions.numpy() + dvf.numpy()`
2. Visualise in 3D Slicer or ITK-SNAP alongside the TLC image.
3. If deformation looks flipped/rotated, try alternative permutations.

### [B] `‖u_sim − u_target‖²` loss (not implemented)

This is the "proper" displacement-mismatch formulation.  It requires:

1. Classify surface nodes: `fem.BoundarySides(geo)` + normal direction check.
2. Build boundary projector for Dirichlet BCs from DIR surface displacements.
3. Assemble 3D NH stiffness matrix K(E).
4. Solve K(E) u_sim = f with `fem_example_utils.bsr_cg`.
5. Record adjoint of linear solve via `tape.record_func(solve_fn, arrays=(rhs, u))`.
6. Compute interior residual: `‖u_sim_interior − u_target_interior‖²`.
7. `tape.backward(loss)` → grad w.r.t. E.

Reference pattern: `warp/examples/fem/example_elastic_shape_optimization.py`
(2D Hookean), adapted to 3D NH on a Tetmesh.

The challenge is that the deformation is large (lung expands ~30% from RV→TLC),
so the linearised stiffness from a single Newton step is inaccurate.  A full
Newton loop is needed inside each optimisation iteration.

### [C] `meshifyVolume()` performance

Pure Python loop over all O(Nx·Ny·Nz) voxels.

| sampling | ~voxels (10123 lung) | typical time |
|----------|----------------------|--------------|
| 8 mm     | ~8 000               | ~5 s         |
| 4 mm     | ~70 000              | ~40 s        |
| 2 mm     | ~500 000             | ~5 min       |

For sampling < 4 mm, consider:
- **TetGen** (`tetgen` Python wrapper): high-quality constrained Delaunay.
- **fTetWild**: robust to non-manifold surfaces, parallel.
- Both produce `.node` / `.ele` files that map to `fem.Tetmesh`.

### [D] Inverted elements (J ≤ 0)

`wp.inverse(F)` produces NaN for degenerate tets.  Mitigation options:

1. **Pre-filter**: discard tets with |det(Dm)| < ε in `meshifyVolume()`.
2. **Regularise F**: replace `F` with `F + ε·I` before inversion.
3. **Alternative energy**: use a Neo-Hookean form that avoids `F^{-T}`,
   e.g. the ARAP or corotational model (less accurate for large strains).

### [E] E parameterisation and Adam stability

After each gradient step, E is clamped to [E_min, E_max] via in-place
`E_field.dof_values.assign(...)`.  This preserves the array object so
Adam's first/second moment buffers remain valid.

For finer spatial resolution (> 10 000 elements) or when E spans many orders
of magnitude, prefer **log-reparameterisation**:

```python
# Replace E_field with theta_field = log(E / E_ref)
theta_field = E_space.make_field()
theta_field.dof_values.fill_(0.0)   # → E = E_ref

# In the integrand:
E_val = E_ref * wp.exp(theta(s))    # always positive, no clamp needed
```

No clamping means Adam momentum is never reset, and the loss landscape
is smoother in log space for materials spanning kPa–MPa ranges.

---

---

## SGD vs Adam for inverse FEM elasticity

### Why Adam was originally chosen
- Adaptive per-parameter learning rates mask wide variation in E across tissue
  types (kPa–MPa), so a single scalar `lr` is sufficient.
- Fast convergence on ill-conditioned problems (heterogeneous curvature).

### Advantages of SGD
| Property | Detail |
|----------|--------|
| Physically interpretable step | Δθ = −lr·g (+ momentum); no hidden per-parameter scaling |
| Smoother E maps | Adam's √v̂ denominator amplifies noisy gradients in low-residual elements, causing speckle artefacts |
| Predictable convergence | Linear rate near minimum; easier to interpret divergence |
| Momentum (β≈0.9) | Accelerates along consistent gradient directions, damps oscillations |
| Nesterov momentum | Corrective look-ahead reduces oscillation in narrow valleys |

### Disadvantages of SGD
- Requires careful `lr` tuning; no per-parameter adaptation.
- Without momentum, convergence is slow (linear rate vs Adam's superlinear-like).

### Recommended lr ranges

| Setting | lr range | Notes |
|---------|----------|-------|
| Vanilla SGD (`momentum=0`) | `1e-4` – `5e-3` | Start at `1e-3`; halve if loss oscillates |
| With momentum (`0.9`) | `1e-4` – `1e-3` | Effective lr ≈ `lr / (1-β)` so use smaller value |
| Nesterov | same as momentum | Generally 10–30% faster than heavy-ball |

Diagnostic: if E collapses to `E_min` after <10 iterations, `lr` is too large.
If loss barely decreases after 50 iterations, `lr` is too small.

---

## HU-based E initialisation

### Physical rationale
Lung CT HU values correlate with local air fraction (and thus tissue density
and stiffness).  At RV, aerated parenchyma is −950 to −800 HU (soft/compliant);
atelectatic / diseased tissue is −200 to +50 HU (stiffer).

### Conversion: `clipped_linear`

    α = clip( (HU − HU_air) / (HU_tissue − HU_air),  0, 1 )
    E = E_init_min + (E_init_max − E_init_min) * α

Then additionally clamped to global `[E_min, E_max]`.

Default: `HU_air = −1000`, `HU_tissue = 0`, `E_init_min = 0.5`, `E_init_max = 20.0`.

### Coordinate system
The RV image is loaded with nibabel (for metadata: shape, zooms, affine), then
**resampled with SimpleITK** to the same `sampling`-mm isotropic grid as the
mask.  Node HU values are sampled by nearest-neighbour lookup using
`ix = round(pos_x / sampling)` — valid because mesh nodes sit exactly on the
resampled grid.

### Outputs when `--save_initial_E_nifti`
| File | Contents |
|------|----------|
| `E_init_volume.nii.gz` | HU-derived initial E, back-projected to image voxel grid |
| `E_volume.nii.gz` | Optimised E (unchanged) |

---

## Key API facts (verified against repo)

| API call | Where verified |
|----------|----------------|
| `fem.Tetmesh(tet_vertex_indices=..., positions=...)` | `example_inv_elast_FEM_usd.py:93` |
| `fem.make_polynomial_space(geo, degree=0, discontinuous=True)` | `example_inv_elast_FEM_usd.py:121` |
| `fem.make_test(space=u_space)` | `example_elastic_shape_optimization.py:187` |
| `fem.integrate(..., output=vec_array)` | `example_elastic_shape_optimization.py:261` |
| `wp.Tape(); tape.backward(loss)` | `example_inv_elast_FEM_usd.py:184-195` |
| `warp.optim.SGD([array], lr, momentum, dampening, nesterov)` | `warp/_src/optim/sgd.py:53`, `example_inv_elast_FEM_usd copy 2.py:119` |
| `warp.optim.Adam([array], lr=...)` | `example_elastic_shape_optimization.py:248` |
| `opt.step([array.grad])` | `example_elastic_shape_optimization.py:333` |
| `wp.squared_norm(mat33)` | `example_inv_elast_FEM_usd.py:153` |
| `wp.atomic_add` in kernel (differentiable) | `example_elastic_shape_optimization.py:123-124` |
| `UsdRenderer.render_mesh` | `example_inv_elast_FEM_usd.py:212-217` |

---

## Files edited / created

| File | Action |
|------|--------|
| `warpbiomechanics/test/inv_elast_lung_v2.py` | **Created** — new script |
| `warpbiomechanics/test/NOTES.md` | **Created** — this file |

Existing scripts in `warpbiomechanics/` are **untouched**.
