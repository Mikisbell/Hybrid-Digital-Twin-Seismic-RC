# Hybrid Digital Twin for Real-Time Seismic Response Prediction of RC Buildings

FEM-Guided Surrogate Modeling for accelerated structural analysis using Physics-Guided Neural Networks (PgNN).

**Target Journal**: [Civil Engineering and Architecture — HRPUB](http://www.hrpub.org/journals/jour_info.php?id=48) (Q2)

## Overview

This framework combines:
- **OpenSeesPy** [1]: High-fidelity nonlinear time history analysis (NLTHA) of N-story RC frames
- **Physics-Guided Neural Networks (PgNN)**: FEM-informed surrogate modeling for peak inter-story drift ratio (IDR) prediction
- **Hybrid Digital Twin** [4]: Coupling high-fidelity FEM with real-time neural inference

The PgNN predicts peak IDR per story from base acceleration, achieving ~2 ms CPU inference latency — a 5000x speedup over full NLTHA (~10 s/record).

### Why Hybrid?

Unlike standard black-box AI, this framework is **Physics-Guided and FEM-Consistent**.
Nonlinear restoring forces $f_{int}$ from OpenSeesPy fiber sections inform per-story
inverse-variance weights during training, encoding concrete cracking, steel yielding,
and cyclic degradation without requiring millions of data points. The architecture
scales from low-rise ($N=3$) to mid-rise ($N=10$) buildings by adjusting the output
dimension.

### FEM-Guided Loss Function

The training objective embeds FEM physics at three levels:

1. **Per-story inverse-variance weights** from NLTHA response statistics
2. **Weighted data fidelity** between predicted and FEM-simulated peak IDR
3. **Physics tensor regularization** via equation-of-motion residual (active in Seq2Seq mode)

$$\mathcal{L}_{total} = \lambda_d\,\mathcal{L}_{data} + \lambda_p\,\mathcal{L}_{reg} + \lambda_b\,\mathcal{L}_{bc}$$

### Validated Results

| Configuration | Records | $R^2$ | RMSE (%) | Latency |
|:---|:---|:---|:---|:---|
| $N=3$ (PEER NGA-West2) | 289 | 0.783 | 0.834 | ~2 ms |
| $N=10$ (PEER NGA-West2) | 265 | 0.713 | — | ~2.5 ms |
| Transfer $N=3 \to 10$ | 265 | 0.700 | — | ~2.5 ms |

## Project Structure

```
Hybrid-Digital-Twin-Seismic-RC/
├── src/                        # Source code
│   ├── config.py               # GlobalConfig — single source of truth
│   ├── opensees_analysis/      # OpenSeesPy RC model & NLTHA runners
│   ├── pinn/                   # Physics-Guided Neural Network
│   │   ├── model.py            # HybridPINN architecture (1D-CNN + Attention)
│   │   ├── loss.py             # FEM-guided composite loss function
│   │   ├── train.py            # Training entry point
│   │   ├── trainer.py          # Training loop & checkpointing
│   │   ├── evaluate.py         # Metrics & publication figures
│   │   └── infer.py            # Quick inference demo
│   ├── preprocessing/          # Data pipeline & feature engineering
│   │   ├── data_factory.py     # PEER AT2 → NLTHA CSV
│   │   └── pipeline.py         # CSV → PyTorch tensors
│   └── utils/                  # Helpers
├── scripts/                    # Utility scripts
│   ├── build_docx.py           # Reproducible Word document generator
│   └── fragility_curves.py     # Fragility curve analysis
├── data/                       # Data storage (heavy files git-ignored)
│   ├── raw/                    # Raw NLTHA simulation output
│   │   ├── peer_3story/        # N=3 campaign (289 records)
│   │   └── peer_10story/       # N=10 campaign (265 records)
│   ├── processed/              # ML-ready PyTorch tensors (.pt)
│   ├── external/               # PEER NGA-West2 ground motions [2]
│   └── models/                 # Trained checkpoints & metrics
├── manuscript/                 # HRPUB paper
│   ├── 01_introduction.md      # Title, abstract, keywords, §1 Introduction
│   ├── 02_objectives.md        # §2 Objectives
│   ├── 03_methods.md           # §3: NLTHA model, PgNN architecture, FEM-guided loss
│   ├── 04_results.md           # §4: Simulation outputs, training, predictions
│   ├── 05_discussion.md        # §5: Interpretation, comparison, whiplash effect
│   ├── 06_conclusions.md       # §6: Key findings, contributions, future work
│   ├── 07_acknowledgements.md  # Acknowledgements
│   ├── references.bib          # BibTeX bibliography — 30 references [1]–[30]
│   └── figures/                # Publication-ready figures (300 DPI, PNG)
├── notebooks/                  # Jupyter notebooks (verification, demos)
├── .github/workflows/          # CI/CD automation
└── requirements.txt            # Python dependencies
```

> **Note**: Heavy data files (`.pt`, `.csv`, model weights) are excluded via
> `.gitignore`. The data pipeline is fully reproducible from source code.
> See `data/README.md` for regeneration instructions.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Mikisbell/Hybrid-Digital-Twin-Seismic-RC.git
cd Hybrid-Digital-Twin-Seismic-RC
```

2. Create virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Activate pre-commit hooks:
```bash
pre-commit install
```

## Getting Started

The three pipeline stages share the `--n-stories` parameter.
`GlobalConfig` (saved by the factory) ensures consistency automatically.

```bash
# 1. Generate NLTHA simulation data (one-time, ~2 min/record)
python src/preprocessing/data_factory.py \
    --n-stories 3 --n-records 300 \
    --output-dir data/raw/peer_3story

# 2. Build ML-ready tensors (validates n_stories against factory config)
python src/preprocessing/pipeline.py \
    --n-stories 3 \
    --raw-dir data/raw/peer_3story --out-dir data/processed/peer_3story

# 3. Train PgNN (n_stories auto-detected from processed data)
python src/pinn/train.py --epochs 500

# 4. Evaluate and generate publication figures
python src/pinn/evaluate.py

# 5. Quick inference demo
python src/pinn/infer.py --save-fig
```

For $N=10$:
```bash
python src/preprocessing/data_factory.py --n-stories 10 --output-dir data/raw/peer_10story
python src/preprocessing/pipeline.py --n-stories 10 --raw-dir data/raw/peer_10story --out-dir data/processed/peer_10story_scalar
python src/pinn/train.py --epochs 1000 --batch-size 16 --processed-dir data/processed/peer_10story_scalar --checkpoint-dir data/models_n10_scalar
```

> **Tip:** If you change `--n-stories`, always restart from Step 1.
> The pipeline raises a `ValueError` if the stored config does not match.

## Features

- Nonlinear dynamic analysis of RC structures (fiber sections, distributed plasticity)
- Parametric N-story surrogate model — validated for $N=3$ and $N=10$
- FEM-guided training objective (inverse-variance weights from $f_{int}$ statistics)
- 1D-CNN encoder with temporal self-attention (4 heads)
- Transfer learning: freeze encoder trained on $N=3$, retrain head for $N=10$
- CPU inference at ~2 ms for real-time structural health monitoring
- Reproducible PEER NGA-West2 pipeline with ASCE 7-22 spectrum scaling
- Centralized `GlobalConfig` preventing n_stories mismatches across pipeline stages

## Requirements

- Python >= 3.10
- OpenSeesPy >= 3.7.0
- PyTorch >= 2.0.0
- NumPy, SciPy, Pandas
- python-docx (for manuscript build)
- pandoc >= 3.0 (for .docx generation)

## Development

### Pre-commit Hooks

Every `git commit` automatically runs: Ruff lint, Ruff format, isort, trailing whitespace, end-of-file fixer, YAML check, JSON check, large file blocker (>1 MB).

### Build Manuscript

```bash
python scripts/build_docx.py
# Output: manuscript/Hybrid_Digital_Twin_Seismic_RC.docx
```

## License

See [LICENSE](LICENSE) for details.

## References

- [1] F. McKenna, "OpenSees: A Framework for Earthquake Engineering Simulation," *Comput. Sci. Eng.*, vol. 13, no. 4, pp. 58-66, 2011.
- [2] T. D. Ancheta *et al.*, "NGA-West2 Database," *Earthquake Spectra*, vol. 30, no. 3, pp. 989-1005, 2014.
- [3] M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural networks," *J. Comput. Phys.*, vol. 378, pp. 686-707, 2019.
- [4] F. Tao *et al.*, "Digital Twin in Industry: State-of-the-Art," *IEEE Trans. Ind. Inform.*, vol. 15, no. 4, pp. 2405-2415, 2019.

## Citation

If you use this framework in your research, please cite:
```
Rivera Ospina, M. A. (2026). Hybrid Digital Twin for Real-Time Seismic Response
Prediction of Reinforced Concrete Buildings Using Physics-Guided Neural Networks.
Civil Engineering and Architecture (HRPUB), under review.
```
