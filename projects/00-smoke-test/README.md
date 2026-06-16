# 00. Environment smoke test

A throwaway sanity check that confirms PyTorch, NumPy, and autograd all work after a fresh `pip install`. The first thing committed to this repo. Not a project. Just a foundation check.

## Run

```bash
.venv/bin/python projects/00-smoke-test/smoke_test.py
```

Expected output: PyTorch version, autograd derivative check (should print `6.0`), and a one-step linear regression loss + gradient.

## Setup

From the repo root:

```bash
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install -r requirements.txt
```

CPU-only build for now. Will revisit when projects need a GPU (probably Month 3-4 with character-level language models).
