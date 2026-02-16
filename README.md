# Hybrid Digital Twin for Seismic RC Buildings

Hybrid Digital Twin framework for real-time seismic damage prediction in RC buildings using OpenSeesPy and Physics-Informed Machine Learning (PIML). Developed for Engineering 4.0 research.

## Overview

This framework combines:
- **OpenSeesPy**: Non-linear time history analysis of RC buildings
- **Physics-Informed Neural Networks (PINNs)**: Real-time structural damage prediction
- **Digital Twin Technology**: Synchronized physical-digital representation for resilience assessment

The system predicts inter-story drifts in real-time, enabling proactive structural health monitoring and seismic risk assessment.

## Project Structure

```
Hybrid-Digital-Twin-Seismic-RC/
├── src/                    # Source code
│   ├── opensees_analysis/  # OpenSeesPy integration modules
│   ├── pinn/              # Physics-Informed Neural Network models
│   ├── preprocessing/     # Data processing utilities
│   └── utils/             # Common utility functions
├── data/                   # Data storage
│   ├── raw/               # Raw OpenSeesPy simulation data
│   ├── processed/         # Processed datasets for ML
│   ├── external/          # External experimental data
│   └── models/            # Trained model checkpoints
├── notebooks/              # Jupyter notebooks
│   ├── Data exploration and visualization
│   ├── PINN development and training
│   └── Digital twin demonstrations
└── requirements.txt        # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Mikisbell/Hybrid-Digital-Twin-Seismic-RC.git
cd Hybrid-Digital-Twin-Seismic-RC
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Getting Started

1. **Run OpenSeesPy Simulations**: Generate structural response data using non-linear time history analysis
2. **Process Data**: Prepare training datasets from simulation results
3. **Train PINN Models**: Develop physics-informed models for damage prediction
4. **Deploy Digital Twin**: Implement real-time prediction system

See individual directory READMEs for detailed information.

## Features

- Non-linear dynamic analysis of RC structures
- Physics-based neural network architectures
- Real-time inter-story drift prediction
- Seismic damage assessment
- Digital twin visualization and monitoring

## Requirements

- Python 3.8+
- OpenSeesPy 3.5+
- PyTorch or TensorFlow
- NumPy, SciPy, Pandas
- Jupyter for notebooks

## License

See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Citation

If you use this framework in your research, please cite:
```
[Citation information to be added]
```
