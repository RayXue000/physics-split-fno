# Physics-Split Fourier Neural Operator

Selected implementation from Rui Xue's complex-plane / physics-informed
operator-learning research. 复平面与物理分裂神经算子的核心代码节选。

The model combines a differentiable known-physics step with a learned
Fourier correction: `G(u) = physics_step(u) + correction(u)`.
This is a physics-split FNO implementation, not a conventional PINN solver.

## Included

- Spectral propagators and differentiable physics steps.
- Equivariant spectral correction and the `PSRFNO` model.
- Amplitude/phase, dissipation/dispersion and data losses.
- Random Fourier-probe JVP response diagnostics.

## Run

Python 3.10+ with a suitable PyTorch installation:
```sh
python -m pip install -r requirements.txt
python example.py
```

The example uses synthetic sinusoidal inputs to check the propagator,
forward/backward pass and rollout. It does not train a model or reproduce
the academic homepage's reported errors.

## Release scope

Core modules are copied from the local `psc_psr_fno` implementation. The
`symbolic` package initializer and synthetic example are release wrappers.
Private datasets, weights, experiment matrices, manuscript drafts, full
training orchestration and unpublished analysis are intentionally omitted.
The research includes conditional and negative results; no universal
accuracy advantage or state-of-the-art claim is made by this code release.

[Research overview and result scope](https://xrnb2000.github.io/)
