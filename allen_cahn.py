import torch
import time
from model import PhaseFieldNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ACPINN:
    def __init__(self, eps: float = 0.15, T: float = 0.5):    
        self.eps = eps
        self.T   = T
        self.model = PhaseFieldNet(layers=[3, 256, 256, 128, 128, 128, 1]).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.999)
        self.history = {"total": [], "pde": [], "ic": [], "bc": []}

    @staticmethod
    def _init_cond(x, y):
        r = torch.sqrt((x - 0.5)**2 + (y - 0.5)**2)
        return torch.tanh((0.25 - r) / 0.15)

    def _pred(self, x, y, t):
        inp = torch.stack([x, y, t], dim=-1)
        return self.model(inp).squeeze(-1)

    def _pde_res(self, x, y, t):
        x.requires_grad_(True); y.requires_grad_(True); t.requires_grad_(True)
        phi = self._pred(x, y, t) 
        
        phi_t  = torch.autograd.grad(phi, t,   grad_outputs=torch.ones_like(phi),  create_graph=True)[0]
        phi_x  = torch.autograd.grad(phi, x,   grad_outputs=torch.ones_like(phi),  create_graph=True)[0]
        phi_y  = torch.autograd.grad(phi, y,   grad_outputs=torch.ones_like(phi),  create_graph=True)[0]
        phi_xx = torch.autograd.grad(phi_x, x, grad_outputs=torch.ones_like(phi_x), create_graph=True)[0]
        phi_yy = torch.autograd.grad(phi_y, y, grad_outputs=torch.ones_like(phi_y), create_graph=True)[0]
        
        return phi_t - self.eps**2 * (phi_xx + phi_yy) + (phi**3 - phi)

    def _loss_pde(self, n=3000):
        x = torch.rand(n, device=device)
        y = torch.rand(n, device=device)
        t = torch.rand(n, device=device) * self.T
        return (self._pde_res(x, y, t)**2).mean() 

    def _loss_ic(self, n=1000):
        x = torch.rand(n, device=device)
        y = torch.rand(n, device=device)
        t = torch.zeros(n, device=device)
        return ((self._pred(x, y, t) - self._init_cond(x, y))**2).mean() 

    def _loss_bc(self, n=500):
        loss = torch.tensor(0.0, device=device)
        t = torch.rand(n, device=device) * self.T
        for edge, coord, var in [
            ("x0", 0.0, "x"), ("x1", 1.0, "x"),
            ("y0", 0.0, "y"), ("y1", 1.0, "y"),
        ]:
            other = torch.rand(n, device=device)
            if "x" in edge:
                x = torch.full((n,), coord, device=device, requires_grad=True)
                y = other
            else:
                y = torch.full((n,), coord, device=device, requires_grad=True)
                x = other
            
            phi = self._pred(x, y, t) 
            ref = x if "x" in edge else y
            dphi_dn = torch.autograd.grad(phi, ref, grad_outputs=torch.ones_like(phi),
                                          create_graph=True)[0]
            loss = loss + (dphi_dn**2).mean()
        return loss / 4

    def train(self, epochs=10000, w_pde=2.0, w_ic=20.0, w_bc=1.5, print_every=1000):
        print("\n" + "="*55)
        print("  Training Allen-Cahn PINN  (ε = {:.3f})".format(self.eps))
        print("="*55)
        t0 = time.time()
        for ep in range(1, epochs + 1):
            self.optimizer.zero_grad()
            l_pde = self._loss_pde()
            l_ic  = self._loss_ic()
            l_bc  = self._loss_bc()
            
            loss  = w_pde * l_pde + w_ic * l_ic + w_bc * l_bc
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.history["total"].append(loss.item())
            self.history["pde"].append(l_pde.item())
            self.history["ic"].append(l_ic.item())
            self.history["bc"].append(l_bc.item())

            if ep % print_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"  Ep {ep:5d} | Total {loss.item():.4e} | "
                      f"PDE {l_pde.item():.4e} | IC {l_ic.item():.4e} | "
                      f"BC {l_bc.item():.4e} | lr {lr:.2e}")
        print(f"  Done in {time.time()-t0:.1f}s")

    @torch.no_grad()
    def pred_grid(self, t_val: float, nx: int = 80):
        xs = torch.linspace(0, 1, nx, device=device)
        ys = torch.linspace(0, 1, nx, device=device)
        xg, yg = torch.meshgrid(xs, ys, indexing="ij")
        tg = torch.full_like(xg, t_val)
        phi = self._pred(xg.reshape(-1), yg.reshape(-1), tg.reshape(-1))
        return phi.reshape(nx, nx).cpu().numpy()

if __name__ == "__main__":
    pinn = ACPINN(eps=0.15, T=0.5)
    pinn.train(epochs=10000)
