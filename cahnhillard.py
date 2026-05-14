##Training using L-BFGS with strong line search ; Spinoidal IC denotes split at interface.
import torch
import torch.nn as nn
import numpy as np
import time
from model import PhaseFieldNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CHPINN:
    def __init__(self, eps: float = 0.05, M: float = 1.0, T: float = 0.5):
        self.eps = eps
        self.M   = M
        self.T   = T
        
        # Output dim is 2 (phi and mu)
        self.model = PhaseFieldNet(layers=[3, 128, 128, 128, 128, 2]).to(device)
        
        self.optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0, 
            max_iter=20, 
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        
        self._last_losses: dict = {}
        self.history = {"total": [], "pde1": [], "pde2": [], "ic": [], "bc": []}

        # --- Pre-generate fixed IC modes ---
        _rng = torch.Generator()
        _rng.manual_seed(7)
        n_modes = 6
        self._ic_kx  = torch.randint(1, 5, (n_modes,), generator=_rng).float().to(device)
        self._ic_ky  = torch.randint(1, 5, (n_modes,), generator=_rng).float().to(device)
        self._ic_amp = (0.3 * (2 * torch.rand(n_modes, generator=_rng) - 1)).to(device)

    def _predict(self, x, y, t):
        inp = torch.stack([x, y, t], dim=-1)
        out = self.model(inp)
        return out[..., 0], out[..., 1]  # returns phi, mu

    def _laplacian(self, f, x, y):
        f_x = torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0]
        f_y = torch.autograd.grad(f, y, grad_outputs=torch.ones_like(f), create_graph=True)[0]
        f_xx = torch.autograd.grad(f_x, x, grad_outputs=torch.ones_like(f_x), create_graph=True)[0]
        f_yy = torch.autograd.grad(f_y, y, grad_outputs=torch.ones_like(f_y), create_graph=True)[0]
        return f_xx + f_yy

    def _initial_condition(self, x, y):
        phi = torch.zeros_like(x)
        for i in range(len(self._ic_kx)):
            phi = phi + self._ic_amp[i] * (
                torch.sin(self._ic_kx[i] * np.pi * x) *
                torch.sin(self._ic_ky[i] * np.pi * y)
            )
        return phi.clamp(-0.95, 0.95)

    def _loss_pde(self, x, y, t):
        x = x.detach().requires_grad_(True)
        y = y.detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)

        phi, mu = self._predict(x, y, t)

        # PDE 1: ∂φ/∂t = M Δμ
        phi_t  = torch.autograd.grad(phi, t, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
        lap_mu = self._laplacian(mu, x, y)
        r1 = phi_t - self.M * lap_mu

        # PDE 2 (constitutive): μ = −ε²Δφ + φ³ − φ
        lap_phi = self._laplacian(phi, x, y)
        r2 = mu - (-self.eps**2 * lap_phi + phi**3 - phi)

        collapse_penalty = torch.relu(0.02 - phi.var())
        return (r1**2).mean(), (r2**2).mean(), collapse_penalty

    def _loss_ic(self, x, y):
        phi_pred, _ = self._predict(x, y, torch.zeros_like(x))
        phi_true = self._initial_condition(x, y)
        return ((phi_pred - phi_true)**2).mean()

    def _loss_bc(self, t, other):
        # Implementation for Neumann (zero-flux) boundary conditions
        loss = 0.0
        for coord in [0.0, 1.0]:
            # X-boundaries
            xb = torch.full_like(other, coord, requires_grad=True)
            phi, mu = self._predict(xb, other, t)
            phi_x = torch.autograd.grad(phi, xb, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
            mu_x  = torch.autograd.grad(mu, xb, grad_outputs=torch.ones_like(mu), create_graph=True)[0]
            loss += (phi_x**2).mean() + (mu_x**2).mean()
            
            # Y-boundaries
            yb = torch.full_like(other, coord, requires_grad=True)
            phi, mu = self._predict(other, yb, t)
            phi_y = torch.autograd.grad(phi, yb, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
            mu_y  = torch.autograd.grad(mu, yb, grad_outputs=torch.ones_like(mu), create_graph=True)[0]
            loss += (phi_y**2).mean() + (mu_y**2).mean()
        return loss / 4

    def train(self, epochs: int = 500, n_pde: int = 800, n_ic: int = 600, n_bc: int = 300,
              w_pde: float = 1.0, w_mu: float = 50.0, w_ic: float = 20.0, 
              w_bc: float = 1.0, w_ac: float = 10.0, print_every: int = 50):

        print("\n" + "="*55)
        print(f"   Training Cahn-Hilliard PINN   (ε={self.eps}, M={self.M})")
        print(f"   Optimiser : L-BFGS   |   epochs = {epochs}")
        print(f"   Weights   : PDE={w_pde}   μ={w_mu}   IC={w_ic}   BC={w_bc}   AC={w_ac}")
        print("="*55)
        t0 = time.time()

        for ep in range(1, epochs + 1):
            x_pde    = torch.rand(n_pde, device=device)
            y_pde    = torch.rand(n_pde, device=device)
            t_pde    = torch.rand(n_pde, device=device) * self.T
            x_ic     = torch.rand(n_ic,  device=device)
            y_ic     = torch.rand(n_ic,  device=device)
            t_bc     = torch.rand(n_bc,  device=device) * self.T
            other_bc = torch.rand(n_bc,  device=device)

            def closure():
                self.optimizer.zero_grad()
                l1, l2, l_ac = self._loss_pde(x_pde, y_pde, t_pde)
                l_ic         = self._loss_ic(x_ic, y_ic)
                l_bc         = self._loss_bc(t_bc, other_bc)
                loss = (w_pde * l1 + w_mu * l2 + w_ic * l_ic + w_bc * l_bc + w_ac * l_ac)
                loss.backward()
                self._last_losses = {
                    "total": loss.item(), "pde1": l1.item(), "pde2": l2.item(),
                    "ic": l_ic.item(), "bc": l_bc.item(), "ac": l_ac.item(),
                }
                return loss

            self.optimizer.step(closure)

            ll = self._last_losses
            self.history["total"].append(ll["total"])
            self.history["pde1"].append(ll["pde1"])
            self.history["pde2"].append(ll["pde2"])
            self.history["ic"].append(ll["ic"])
            self.history["bc"].append(ll["bc"])

            if ep % print_every == 0 or ep == 1:
                print(f"   Ep {ep:4d} | Total {ll['total']:.4e} | "
                      f"φ_t {ll['pde1']:.4e} | μ {ll['pde2']:.4e} | "
                      f"IC {ll['ic']:.4e} | AC {ll['ac']:.4e} | "
                      f"t {time.time()-t0:.1f}s")

        print(f"\n   Done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    pinn = CHPINN()
    pinn.train()
