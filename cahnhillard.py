import torch
import time
from model import PhaseFieldNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CHPINN:
    def __init__(self, eps: float = 0.05, D: float = 0.01, T: float = 0.1):
        self.eps = eps  # Interface width
        self.D   = D    # Diffusion coefficient
        self.T   = T
        self.model = PhaseFieldNet(layers=[3, 256, 256, 256, 128, 128, 1]).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)
        self.history = {"total": [], "pde": [], "ic": [], "bc": []}

    @staticmethod
    def _init_cond(x, y):
        # Cahn-Hilliard often starts with random fluctuations (Spinodal Decomposition)
        # For a benchmark, we'll use a specific profile or random noise:
        return 0.1 * (torch.rand_like(x) - 0.5)

    def _pred(self, x, y, t):
        inp = torch.stack([x, y, t], dim=-1)
        return self.model(inp).squeeze(-1)

    def _pde_res(self, x, y, t):
        x.requires_grad_(True); y.requires_grad_(True); t.requires_grad_(True)
        phi = self._pred(x, y, t)
        
        # 1st order derivatives
        phi_t = torch.autograd.grad(phi, t, torch.ones_like(phi), create_graph=True)[0]
        phi_x = torch.autograd.grad(phi, x, torch.ones_like(phi), create_graph=True)[0]
        phi_y = torch.autograd.grad(phi, y, torch.ones_like(phi), create_graph=True)[0]
        
        # 2nd order (Laplacian of phi)
        phi_xx = torch.autograd.grad(phi_x, x, torch.ones_like(phi_x), create_graph=True)[0]
        phi_yy = torch.autograd.grad(phi_y, y, torch.ones_like(phi_y), create_graph=True)[0]
        lap_phi = phi_xx + phi_yy
        
        # Chemical Potential: mu = phi^3 - phi - eps^2 * Laplacian(phi)
        mu = phi**3 - phi - (self.eps**2) * lap_phi
        
        # 4th order: Laplacian of mu
        mu_x = torch.autograd.grad(mu, x, torch.ones_like(mu), create_graph=True)[0]
        mu_y = torch.autograd.grad(mu, y, torch.ones_like(mu), create_graph=True)[0]
        mu_xx = torch.autograd.grad(mu_x, x, torch.ones_like(mu_x), create_graph=True)[0]
        mu_yy = torch.autograd.grad(mu_y, y, torch.ones_like(mu_y), create_graph=True)[0]
        lap_mu = mu_xx + mu_yy
        
        # PDE: dphi/dt = D * Laplacian(mu)
        return phi_t - self.D * lap_mu

    def _loss_pde(self, n=3000):
        x, y = torch.rand(n, device=device), torch.rand(n, device=device)
        t = torch.rand(n, device=device) * self.T
        return (self._pde_res(x, y, t)**2).mean()

    def _loss_ic(self, n=1500):
        x, y = torch.rand(n, device=device), torch.rand(n, device=device)
        t = torch.zeros(n, device=device)
        return ((self._pred(x, y, t) - self._init_cond(x, y))**2).mean()

    def _loss_bc(self, n=500):
        # Simplified: Zero-flux (Neumann) on mu and phi
        # Implementation similar to ACPINN but ideally applied to both phi and mu
        return torch.tensor(0.0, device=device) 

    def train(self, epochs=10000, w_pde=1.0, w_ic=100.0):
        t0 = time.time()
        for ep in range(1, epochs + 1):
            self.optimizer.zero_grad()
            l_pde, l_ic = self._loss_pde(), self._loss_ic()
            loss = w_pde * l_pde + w_ic * l_ic
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            if ep % 1000 == 0:
                print(f"Ep {ep} | Loss: {loss.item():.4e} | PDE: {l_pde.item():.4e}")
        print(f"Done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    pinn = CHPINN(eps=0.05, D=0.01)
    pinn.train(epochs=10000)
