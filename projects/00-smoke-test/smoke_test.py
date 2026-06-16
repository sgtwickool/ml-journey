"""Confirms the local Python/PyTorch environment is alive.

Run from the repo root:
    .venv/bin/python projects/00-smoke-test/smoke_test.py
"""

import sys
import platform

import torch
import numpy as np


def main() -> None:
    print(f"Python:       {platform.python_version()}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"NumPy:        {np.__version__}")
    print(f"CUDA avail:   {torch.cuda.is_available()}")
    print(f"MPS avail:    {torch.backends.mps.is_available()}")
    print()

    # Tiny autograd check: f(x) = x^2, df/dx at x=3 should be 6.
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2
    y.backward()

    assert x.grad is not None and abs(x.grad.item() - 6.0) < 1e-6, "autograd broken"
    print(f"autograd OK:  d(x^2)/dx at x=3 -> {x.grad.item()}")

    # Tiny linear regression by hand, one gradient step.
    w = torch.tensor(0.0, requires_grad=True)
    xs = torch.tensor([1.0, 2.0, 3.0, 4.0])
    ys = torch.tensor([2.0, 4.0, 6.0, 8.0])  # true w = 2

    pred = w * xs
    loss = ((pred - ys) ** 2).mean()
    loss.backward()

    print(f"loss before:  {loss.item():.4f}  grad: {w.grad.item():.4f}")
    print("env is ready for micrograd.")


if __name__ == "__main__":
    sys.exit(main())
