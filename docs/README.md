# Documentation

Quick index for this repository (Part II dissertation: machine unlearning on CIFAR-10 with ResNet-50).

## Start here

| Document | What it covers |
|----------|----------------|
| [project-overview.md](project-overview.md) | What the pipeline does, main code paths, notebook flow |
| [../src/README.md](../src/README.md) | `gradient_ascent` package module map |
| [evaluation/multitarget-averaging.md](evaluation/multitarget-averaging.md) | Averaging results over all 10 forget classes; output files |
| [evaluation/similarity-mia-correlation.md](evaluation/similarity-mia-correlation.md) | Scalar similarity change vs MIA reduction (exploratory correlation) |
| [evaluation/membership-inference.md](evaluation/membership-inference.md) | MIA protocol: probes, CV, attackers, mapping to prior work |
| [methods/ssd-and-scrub.md](methods/ssd-and-scrub.md) | Why SCRUB is included; SSD/SCRUB runtime fixes and fair sweeps |
| [similarity/visualisation-and-performance.md](similarity/visualisation-and-performance.md) | Similarity plots (GIFs, before/after) and speed optimisations |
| [engineering/reproducibility-and-data.md](engineering/reproducibility-and-data.md) | CIFAR-10 download mirrors; Colab checkpoint reuse |
| [engineering/codebase-notes.md](engineering/codebase-notes.md) | Core pipeline pruning and notebook extraction (maintenance) |

## Folder layout

```
docs/
├── README.md                    ← you are here
├── project-overview.md          (also: ../src/README.md for code map)
├── evaluation/
│   ├── multitarget-averaging.md
│   ├── similarity-mia-correlation.md
│   └── membership-inference.md
├── methods/
│   └── ssd-and-scrub.md
├── similarity/
│   └── visualisation-and-performance.md
└── engineering/
    ├── reproducibility-and-data.md
    └── codebase-notes.md
```

Older one-topic files in the repo root `docs/` were merged into these paths; use the table above instead of dated filenames.
