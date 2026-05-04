# gradient-ascent

Part II project: **machine unlearning** on **CIFAR-10** with **ResNet-50** (PyTorch). Training, five unlearning baselines, retrain-from-scratch references, **similarity trajectories**, and **membership inference** evaluation.

## Documentation

Start at **[docs/README.md](docs/README.md)** for methodology and engineering notes.

## Repository layout

| Path | Purpose |
|------|---------|
| `src/gradient_ascent/` | Installable package (see **[src/README.md](src/README.md)** for a module map) |
| `notebooks/` | Main experiment notebook |
| `tests/` | Unit/smoke tests |
| `docs/` | Written methodology and design notes |

## Running

- Main workflow: `notebooks/experiments.ipynb`
- Package: `pip install -e .` from repo root, then `import gradient_ascent`

Install and dependencies: see `pyproject.toml` / `requirements.txt`.
