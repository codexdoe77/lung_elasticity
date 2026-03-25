import numpy as np
import warp as wp
import warp.fem as fem
import warp.examples.fem.utils as fem_utils

wp.init()

@fem.integrand
def nh_energy_integrand(s: fem.Sample, u: fem.Field, mu: fem.Field):
    # Compute deformation gradient
    F = wp.identity(n=2, dtype=float) + fem.grad(u, s)
    J = wp.determinant(F)
    dJ_dF = wp.mat22(F[1, 1], -F[1, 0], -F[0, 1], F[0, 0])

    mu_val = mu(s)
    lam_val = 100.0  # fixed lambda

    Psi = mu_val * wp.squared_norm(F) + lam_val * (J - 1.0 - mu_val / lam_val)**2
    return Psi

class InverseElasticityExample:
    def __init__(self, resolution=25, degree=2):
        self.geo = fem.Grid2D(res=wp.vec2i(resolution))
        self.u_space = fem.make_polynomial_space(self.geo, degree=degree, dtype=wp.vec2)
        self.mu_space = fem.make_polynomial_space(self.geo, degree=1, dtype=float)

        self.u_field = self.u_space.make_field()
        self.mu_field = self.mu_space.make_field()

        # Load known displacement (mock for now)
        np.random.seed(42)
        self.u_field.dof_values[:] = wp.array(
            np.random.uniform(-0.05, 0.05, size=self.u_field.dof_values.shape),
            dtype=wp.vec2,
        )
        self.mu_field.dof_values.fill_(1.0)  # initial guess

        self.domain = fem.Cells(self.geo)

    def compute_loss_and_grad(self):
        with wp.ScopedTape() as tape:
            tape.watch(self.mu_field.dof_values)
            loss = fem.integrate(
                nh_energy_integrand,
                fields={"u": self.u_field, "mu": self.mu_field},
                domain=self.domain,
            )
        grad = tape.grad(self.mu_field.dof_values)
        return loss, grad

    def optimize(self, iterations=20, lr=0.1):
        for i in range(iterations):
            loss, grad = self.compute_loss_and_grad()
            self.mu_field.dof_values -= lr * grad
            print(f"Iter {i}: Loss = {loss:.6f}, Grad Norm = {wp.norm(grad):.6f}")

if __name__ == "__main__":
    example = InverseElasticityExample(resolution=25)
    example.optimize(iterations=20, lr=0.1)
