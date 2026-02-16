# Data Directory

This directory contains all datasets used for training and validating the Physics-Informed Neural Network.

## Structure

Suggested subdirectories:

- **raw/**: Raw data from OpenSeesPy simulations
  - Time history data
  - Ground motion records
  - Structural response data

- **processed/**: Processed and cleaned datasets
  - Normalized inter-story drift data
  - Feature-engineered datasets
  - Training/validation/test splits

- **external/**: External datasets from experimental tests or other sources
  - Shake table test data
  - Field monitoring data
  - Benchmark datasets

- **models/**: Saved PINN model checkpoints and weights

## Data Format

Data files should be stored in standard formats:
- CSV for tabular data
- HDF5 or NPY for large numerical arrays
- JSON for metadata and configuration files

## Note

Large data files (>100 MB) should not be committed to Git. Consider using Git LFS or external storage solutions for large datasets.
