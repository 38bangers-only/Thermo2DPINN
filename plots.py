import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

# --- Branch Imports ---
try:
    from allencahn import ACPINN
    from cahnhilliard import CHPINN
except ImportError:
    print("Error: Ensure allencahn.py and cahnhilliard.py are in the same directory.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_results(ac, ch):
    T = ac.T
    t_snapshots = [0.0, T * 0.25, T * 0.5, T]
    nx = 80

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#0d0d0d")
    

    gs = gridspec.GridSpec(4, 6, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.06, right=0.97, top=0.92, bottom=0.06)

    # Header
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(0.5, 0.7, "2D Phase-Field PINNs · Allen-Cahn & Cahn-Hilliard",
                  ha="center", va="center", fontsize=17, color="white",
                  fontfamily="monospace", fontweight="bold")
    ax_title.text(0.5, 0.15, f"ε = {ac.eps}  |  domain [0,1]²  |  T = {T}  |  Neumann BC",
                  ha="center", va="center", fontsize=10, color="#aaaaaa", fontfamily="monospace")

    # Spatial Snapshots
    kw_ac = dict(cmap="RdBu_r", vmin=-1.05, vmax=1.05, origin="lower", extent=[0, 1, 0, 1])
    kw_ch = dict(cmap="PiYG", vmin=-1.0, vmax=1.0, origin="lower", extent=[0, 1, 0, 1])

    for col, t_val in enumerate(t_snapshots):
        # Allen-Cahn
        phi_ac = ac.pred_grid(t_val, nx)
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(phi_ac.T, **kw_ac)
        ax.set_title(f"t = {t_val:.3f}", color="#cccccc", fontsize=9, fontfamily="monospace")
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0: ax.set_ylabel("Allen-Cahn φ", color="#ff9966", fontsize=9)

        # Cahn-Hilliard
        phi_ch, mu_ch = ch.pred_grid(t_val, nx)
        
        ax2 = fig.add_subplot(gs[2, col])
        ax2.imshow(phi_ch.T, **kw_ch)
        ax2.set_xticks([]); ax2.set_yticks([])
        if col == 0: ax2.set_ylabel("Cahn-Hilliard φ", color="#66ccff", fontsize=9)

        ax3 = fig.add_subplot(gs[3, col])
        im3 = ax3.imshow(mu_ch.T, cmap="plasma", origin="lower", extent=[0, 1, 0, 1])
        ax3.set_xticks([]); ax3.set_yticks([])
        if col == 0: ax3.set_ylabel("Cahn-Hilliard μ", color="#cc99ff", fontsize=9)

    # Loss Diagnostics
    ax_l1 = fig.add_subplot(gs[1, 4:])
    ax_l2 = fig.add_subplot(gs[2, 4:])
    ax_l3 = fig.add_subplot(gs[3, 4:])

    def _plot_loss(ax, history, keys, colors, title):
        for k, c in zip(keys, colors):
            if k in history and history[k]:
                ax.semilogy(history[k], color=c, lw=1.2, label=k)
        ax.set_facecolor("#1a1a1a")
        ax.set_title(title, color="#cccccc", fontsize=9, fontfamily="monospace")
        ax.tick_params(colors="#888")
        ax.legend(fontsize=7, labelcolor="white", facecolor="#111", loc="upper right")

    _plot_loss(ax_l1, ac.history, ["total", "pde", "ic", "bc"],
               ["#ff9966", "#ff4444", "#44ff88", "#4488ff"], "Allen-Cahn Training")
    
    _plot_loss(ax_l2, ch.history, ["total", "pde1", "pde2", "ic"],
               ["#66ccff", "#ff4444", "#ffaa44", "#44ff88"], "Cahn-Hilliard Training")

    # Physics Diagnostics
    t_vals = np.linspace(0, T, 20)
    energies_ac, mass_ch = [], []
    for tv in t_vals:
        p_ac = ac.pred_grid(tv, 40)
        energies_ac.append((1.0/39)**2 * np.sum((p_ac**2 - 1)**2 / 4))
        p_ch, _ = ch.pred_grid(tv, 40)
        mass_ch.append(p_ch.mean())

    ax_l3.set_facecolor("#1a1a1a")
    ax_l3_r = ax_l3.twinx()
    ax_l3.plot(t_vals, energies_ac, color="#ff9966", lw=1.5, label="AC Energy")
    ax_l3_r.plot(t_vals, mass_ch, color="#66ccff", lw=1.5, ls="--", label="CH Mass")
    ax_l3.set_title("Conservation & Energy", color="#cccccc", fontsize=9)
    ax_l3.tick_params(colors="#888"); ax_l3_r.tick_params(colors="#888")
    ax_l3.legend(loc="upper left", fontsize=7); ax_l3_r.legend(loc="upper right", fontsize=7)

    plt.savefig("results_dashboard.png", dpi=300, facecolor="#0d0d0d")
    plt.show()

if __name__ == "__main__":
    T, EPS = 0.3, 0.05

    # 1. Train Allen-Cahn
    ac_solver = ACPINN(eps=EPS, T=T)
    ac_solver.train(epochs=10000, print_every=1000)

    # 2. Train Cahn-Hilliard
    ch_solver = CHPINN(eps=EPS, M=1.0, T=T)
    ch_solver.train(epochs=500, print_every=50) 

    # 3. Save and Plot
    torch.save(ac_solver.model.state_dict(), "ac_final.pth")
    torch.save(ch_solver.model.state_dict(), "ch_final.pth")
    plot_results(ac_solver, ch_solver)
