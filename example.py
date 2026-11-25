"""Synthetic forward/backward and propagator checks, not a trained benchmark."""
import math
import torch
from models import PSRFNO, AdvectionDiffusionStep

torch.manual_seed(0)
grid = torch.arange(64, dtype=torch.float32) / 64
u = torch.sin(2 * math.pi * grid).repeat(2, 1)
dt, nu = 0.01, 0.02
step = AdvectionDiffusionStep(velocity=0.0, nu=nu, dt=dt)
exact = math.exp(-nu * (2 * math.pi) ** 2 * dt) * u
assert torch.allclose(step(u), exact, atol=2e-6)
model = PSRFNO(known_step=step, modes=8, width=16, n_layers=2)
y = model(u)
assert y.shape == u.shape and torch.isfinite(y).all()
y.square().mean().backward()
assert any(p.grad is not None for p in model.parameters())
assert model.rollout(u, 3).shape == (4, 2, 64)
print("PASS: analytic diffusion, forward/backward, rollout shapes")
