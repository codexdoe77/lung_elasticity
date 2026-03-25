import numpy as np
import warp as wp
import warp.fem as fem
import cudaSetConnect
import SimpleITK as sitk

from warp.render.render_usd import UsdRenderer
# from warp.render.render_usd import save

# import warp.sim.render
# from warp.examples.fem.utils import Plot

import warp.optim
wp.init()

# ---- USER INPUT ----
# mask_path = "/path/to/lung_mask.nii.gz"
# u_path = "/path/to/U_mat.nii.gz"
# v_path = "/path/to/V_mat.nii.gz"
# w_path = "/path/to/W_mat.nii.gz"
mask_path="/home/marvel/Documents/june2025/6-26-25/matlab/10123_002BERFR_uncropped/niftis/10123_002BERFR_RV_maskLobar_iso1.0_uncrop.nii.gz"
u_path="/home/marvel/Documents/july2025/7-2-25/U_mat.nii.gz"
v_path="/home/marvel/Documents/july2025/7-2-25/V_mat.nii.gz"
w_path="/home/marvel/Documents/july2025/7-2-25/W_mat.nii.gz"

sampling = 8.0  # mm
poisson_ratio = 0.43  # Fixed for all elements

# ---- LOAD AND RESAMPLE ----
def make_isotropic(image, sampling=1):
    spacing = [sampling] * image.GetDimension()
    size = [int(round(osz * ospc / nspc + 0.3)) for osz, ospc, nspc in zip(image.GetSize(), image.GetSpacing(), spacing)]
    return sitk.Resample(image, size, sitk.Transform(), sitk.sitkLinear, image.GetOrigin(), spacing, image.GetDirection(), 0, image.GetPixelID())

mask = make_isotropic(sitk.ReadImage(mask_path), sampling=sampling)
u_img = make_isotropic(sitk.ReadImage(u_path), sampling=sampling)
v_img = make_isotropic(sitk.ReadImage(v_path), sampling=sampling)
w_img = make_isotropic(sitk.ReadImage(w_path), sampling=sampling)

# ---- GET VOLUME & SPACING ----
volume = sitk.GetArrayFromImage(mask).astype(np.uint8)  # z, y, x
volume = np.transpose(volume, (2, 1, 0))  # x, y, z
spacing = list(mask.GetSpacing())[::-1]  # x, y, z

# ---- BUILD MESH ----
# connector = cudaSetConnect.SetConnect(arraySize=volume.shape, iso_resolution=sampling, volume=volume)
# positions = connector.points.astype(np.float32)
# tets = connector.tetrahedra.astype(np.int32)
# ---- BUILD MESH ----
# connector = cudaSetConnect.SetConnect(arraySize=volume.shape, iso_resolution=sampling, volume=volume)
connector = cudaSetConnect.SetConnect(arraySize=volume.shape, iso_resolution=sampling, volume=volume.flatten(order='F'))
connector.findConnections()
connector.meshifyVolume()
positions=wp.array(connector.getVolumePoints(),dtype=wp.vec3)
# positions = connector.points.astype(np.float32)
# tets = connector.tetrahedra.astype(np.int32)
tets=wp.array(connector.tetras.astype(np.int32))
geo = fem.Tetmesh(tet_vertex_indices=tets, positions=positions)

# # def deform(positions: wp.array(dtype=wp.vec3), t: float):

# 
# ---- LOAD DVF ----
# .flatten(order='F')
u = np.transpose(sitk.GetArrayFromImage(u_img), (2, 1, 0)).flatten(order='F')
v = np.transpose(sitk.GetArrayFromImage(v_img), (2, 1, 0)).flatten(order='F')
w = np.transpose(sitk.GetArrayFromImage(w_img), (2, 1, 0)).flatten(order='F')

du = u[connector.nzindices]
dv = v[connector.nzindices]
dw = w[connector.nzindices]
# du = ucrop[connector.nzindices] * 0.001
# dv = vcrop[connector.nzindices] * 0.001
# dw = wcrop[connector.nzindices] * 0.001
dvf = wp.array(np.stack([dw,du,dv], axis=1),dtype=wp.vec3)



# dvf = np.stack([u, v, w], axis=1).astype(np.float32)
assert dvf.shape[0] == positions.shape[0], "DVF and mesh size mismatch"

gt_pos = positions + dvf

# ---- SPACES ----
u_space = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec3)
E_space = fem.make_polynomial_space(geo, degree=0, dtype=float, discontinuous=True)

u_field = u_space.make_field()
E_field = E_space.make_field()

u_field.dof_values.assign(gt_pos - positions)  # displacement field
E_field.dof_values.fill_(1.0)  # initial guess for Young's modulus

# ---- ENERGY INTEGRAND ----
@fem.integrand
def nh_energy_young(s: fem.Sample, u: fem.Field, E: fem.Field, nu: float):
    F = wp.identity(3) + fem.grad(u, s)
    J = wp.determinant(F)

    E_val = E(s)[0]
    mu = E_val / (2.0 * (1.0 + nu))
    lam = E_val * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    dJ_dF = wp.mat33(
        F[1, 1]*F[2, 2] - F[1, 2]*F[2, 1],
        F[0, 2]*F[2, 1] - F[0, 1]*F[2, 2],
        F[0, 1]*F[1, 2] - F[0, 2]*F[1, 1],
        F[1, 2]*F[2, 0] - F[1, 0]*F[2, 2],
        F[0, 0]*F[2, 2] - F[0, 2]*F[2, 0],
        F[0, 2]*F[1, 0] - F[0, 0]*F[1, 2],
        F[1, 0]*F[2, 1] - F[1, 1]*F[2, 0],
        F[0, 1]*F[2, 0] - F[0, 0]*F[2, 1],
        F[0, 0]*F[1, 1] - F[0, 1]*F[1, 0]
    )

    energy = mu * wp.sqr(F) + lam * (J - 1.0 - mu / lam)**2
    return energy

# ---- OPTIMIZATION ----
# opt = wp.optim.SGD(lr=1e-2)

# for it in range(100):
#     with wp.ScopedDevice() as tape:
#         tape.watch(E_field.dof_values)
#         loss = fem.integrate(nh_energy_young, domain=fem.Cells(geo),
#                              fields={"u": u_field, "E": E_field},
#                              values={"nu": poisson_ratio})
#     print(f"Iter {it}, Loss: {loss.numpy():.6f}")
#     opt.step(E_field.dof_values.grad)



# 1. Init
renderer = UsdRenderer("fem_mesh_7-9.usd")

# 2. Extract geometry
positions = geo.positions.numpy()
tets = geo.tet_vertex_indices.numpy()

# 3. Convert tets to surface triangles
faces = []
for tet in tets:
    i, j, k, l = tet
    faces += [[i, j, k], [i, j, l], [i, k, l], [j, k, l]]

# 4. Render mesh
renderer.render_mesh(
    name="fem_surface",
    points=positions,
    indices=faces,
    # colors=[(0.2, 0.6, 0.9)] * len(positions)
)

# 5. Save
renderer.save()