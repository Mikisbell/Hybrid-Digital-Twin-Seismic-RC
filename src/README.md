# Source Code — Hybrid Digital Twin Framework

## Structure

```
src/
├── config.py                    # GlobalConfig: n_stories, n_bays, seq_len, dt
├── __init__.py
├── opensees_analysis/
│   └── ospy_model.py            # Parametric N-story RC frame (Concrete02 + Steel02)
├── pinn/
│   ├── model.py                 # HybridPINN: 1D-CNN (32→64→128) + Temporal Attention (4 heads) + FC head
│   ├── loss.py                  # FEM-guided composite loss (L_data + L_reg + L_bc)
│   ├── train.py                 # Training entry point (auto-detects n_stories from GlobalConfig)
│   ├── trainer.py               # Training loop, checkpointing, early stopping
│   ├── evaluate.py              # Test metrics + publication figures (Figs 4–7)
│   └── infer.py                 # Quick inference demo (AT2 or test.pt input)
├── preprocessing/
│   ├── data_factory.py          # PEER AT2 → OpenSeesPy NLTHA → CSV (saves GlobalConfig)
│   └── pipeline.py              # CSV → PyTorch tensors (validates GlobalConfig)
└── utils/
    └── ...                      # Notion sync, figure manager, helpers
```

## Key Patterns

### GlobalConfig (`config.py`)
Single source of truth for pipeline consistency. The factory saves it; the pipeline
validates against it; the trainer auto-detects `n_stories` from it.

### Physics-Guided Loss (`pinn/loss.py`)
Three-level FEM integration:
1. Per-story inverse-variance weights from NLTHA $f_{int}$ statistics
2. Weighted MSE between predicted and FEM-simulated peak IDR
3. Equation-of-motion residual (active in Seq2Seq mode; indicator-only in scalar mode)

### Output Modes
- **Scalar**: Predicts peak IDR per story → shape `(B, N_stories)`
- **Seq2Seq**: Predicts full displacement time history → shape `(B, N_stories, T)`
