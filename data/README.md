# Data Directory

All datasets for training and validating the Physics-Guided Neural Network.
Heavy files (`.pt`, `.csv`, model weights) are excluded via `.gitignore`.

## Structure

```
data/
├── raw/                         # Raw NLTHA simulation output (CSV + JSON metadata)
│   ├── peer_3story/             # N=3 campaign: 289 converged records
│   └── peer_10story/            # N=10 campaign: 265 converged records
├── processed/                   # ML-ready PyTorch tensors
│   ├── peer_3story/             # train.pt, val.pt, test.pt (N=3, Seq2Seq)
│   ├── peer_3story_seq/         # N=3 sequence variant
│   ├── peer_10story_scalar/     # train.pt, val.pt, test.pt (N=10, scalar)
│   └── peer_10story_seq/        # N=10 sequence variant
├── external/                    # PEER NGA-West2 ground motions (.AT2)
│   └── peer_nga/                # Raw AT2 files (see peer_nga/README.md)
├── models/                      # N=3 trained checkpoints
│   ├── pinn_best.pt             # Best model (by val loss)
│   ├── train_history.json       # Loss curves
│   ├── test_metrics.json        # N=3 test set metrics
│   └── benchmark_results.json   # Latency benchmarks
├── models_n10_scalar/           # N=10 scalar model
│   ├── pinn_best.pt
│   ├── train_history.json
│   └── test_metrics.json
└── models_n10_transfer/         # N=3→N=10 transfer learning model
```

## Data Format

| File | Format | Description |
|:---|:---|:---|
| `raw/*/*.csv` | CSV | Per-record NLTHA output (time, accel, displacements, f_int) |
| `raw/*/*_meta.json` | JSON | Record metadata (RSN, Mw, Rjb, Vs30, scale factor) |
| `processed/*/train.pt` | PyTorch | `{'X': (B,1,T), 'y': (B,N), 'physics': {...}}` |
| `models/*/pinn_best.pt` | PyTorch | Model state dict + config |
| `global_config.json` | JSON | GlobalConfig snapshot (n_stories, seq_len, dt) |

## Regeneration

```bash
# N=3 full pipeline
python src/preprocessing/data_factory.py --n-stories 3 --output-dir data/raw/peer_3story
python src/preprocessing/pipeline.py --n-stories 3 --raw-dir data/raw/peer_3story --out-dir data/processed/peer_3story
python src/pinn/train.py --epochs 500

# N=10 scalar pipeline
python src/preprocessing/data_factory.py --n-stories 10 --output-dir data/raw/peer_10story
python src/preprocessing/pipeline.py --n-stories 10 --raw-dir data/raw/peer_10story --out-dir data/processed/peer_10story_scalar
python src/pinn/train.py --epochs 1000 --batch-size 16 --processed-dir data/processed/peer_10story_scalar --checkpoint-dir data/models_n10_scalar
```

> **Important**: Always pass `--n-stories` consistently. The pipeline validates
> against the `global_config.json` saved by the factory and raises `ValueError`
> on mismatch.
