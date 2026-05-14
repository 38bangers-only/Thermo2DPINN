# Thermo2DPINN
# Thermo2DPINN

2D  Thermodynamic Physics-Informed Neural Networks (TPINNs) for the Allen-Cahn and Cahn-Hilliard phase-field equations, implemented in PyTorch.

---

## Equations

**Allen-Cahn**
$$\frac{\partial \phi}{\partial t} = \varepsilon^2 \nabla^2 \phi - (\phi^3 - \phi)$$

**Cahn-Hilliard**
$$\frac{\partial \phi}{\partial t} = M \nabla^2 \mu, \quad \mu = -\varepsilon^2 \nabla^2 \phi + \phi^3 - \phi$$

Both equations are solved on the unit domain $[0,1]^2$ with Neumann (zero-flux) boundary conditions.

---

## Architecture

- Fully-connected network with **Random Fourier Feature** embedding (fixed frequency matrix, σ=1.0)
- 5 hidden layers, Tanh activations, Xavier initialisation
- Allen-Cahn: single output (φ), optimised with **Adam + ExponentialLR**
- Cahn-Hilliard: two outputs (φ, μ), optimised with **L-BFGS + strong line search**

---

## Key Design Decisions

**Fixed-batch closure for L-BFGS**
Collocation points are sampled once per outer iteration and held fixed inside the closure. L-BFGS performs up to 20 internal line-search evaluations per step — re-sampling inside the closure would present a different loss surface on each sub-step, violating the line-search assumptions and degrading convergence.

**Structured IC to prevent trivial collapse**
The Cahn-Hilliard equation admits φ=0, μ=0 as an exact trivial solution. A random noise IC centred at 0 makes this easy for the network to find. Instead, the initial condition is a fixed sum of Fourier sine modes with amplitude ~0.3, generated once at initialisation using a private RNG so every closure call sees the identical target field.

**Anti-collapse penalty**
An additional penalty term `relu(0.02 - var(φ))` is added to the loss. It is zero when the field has meaningful spatial variation and positive only when the network is collapsing toward a uniform state.

**No LR scheduler with L-BFGS**
ExponentialLR is incompatible with L-BFGS because the scheduler would fire up to `max_iter` times per outer step rather than once per epoch.

---

## Results

Trained on a single NVIDIA T4 GPU (Google Colab). Parameters: ε=0.15, M=1.0, T=0.5.
<img width="5706" height="3792" alt="results_dashboard" src="https://github.com/user-attachments/assets/4005bea9-be61-4096-aa46-3f71dd12c692" />


Allen-Cahn shows smooth interface evolution of a circular droplet. Cahn-Hilliard shows characteristic spinodal decomposition with a bicontinuous morphology. The CH mass is approximately conserved across time snapshots.

---

## Limitations

PINNs encode the full space-time solution in a single network, which makes long-time integration (T > 0.5) unreliable for these equations. The network capacity required to resolve sharp interfaces over long timescales grows substantially, and accuracy degrades. This is a known limitation of the PINN formulation for phase-field PDEs and was the primary motivation for exploring operator learning approaches (FNO, DeepONet) for this class of problems.

---

## Repository Structure

```
Thermo2DPINN/
├── model.py          # PhaseFieldNet architecture with Fourier features
├── allen_cahn.py     # AllenCahnPINN class (Adam)
├── cahnhilliard.py   # CahnHilliardPINN class (L-BFGS)
├── utils.py          # Visualisation and physics diagnostics
├── Dockerfile        # Containerised training pipeline
├── requirements.txt  # Python dependencies
└── results/          # T4 output figures
```

---

## Usage

**Local**
```bash
pip install -r requirements.txt
python utils.py
```

**Docker**
```bash
docker build -t thermo2dpinn .
docker run -v $(pwd)/results:/app/results thermo2dpinn
```

---

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| ε | 0.15 | Interface width |
| M | 1.0 | Cahn-Hilliard mobility |
| T | 0.5 | Simulation end time |
| Layers | [3,256,256 128, 128, 128,1/2] | Network depth and width |
| Fourier frequencies | 32 | Random Fourier feature dimension |
| L-BFGS max_iter | 20 | Max line-search steps per outer iteration |
| L-BFGS history | 50 | Curvature history size |

---

## Dependencies

- Python 3.10+
- PyTorch 2.3.0
- NumPy
- Matplotlib

---

## Notes

This codebase was developed independently as an exploration of PINN-based solvers for thermodynamic phase-field equations. It is not associated with any ongoing research publication.
