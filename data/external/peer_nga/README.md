# PEER NGA-West2 Ground Motion Records

This directory stores the raw `.AT2` acceleration files downloaded from the
[PEER NGA-West2 database](https://ngawest2.berkeley.edu/).

## Download Instructions

### Step 1: Access the Database

1. Go to <https://ngawest2.berkeley.edu/>
2. Create a free account (or log in).
3. Navigate to **Search & Download** → **Advanced Search**.

### Step 2: Apply Selection Criteria (Manuscript §3.2)

| Parameter | Range | Rationale |
|-----------|-------|-----------|
| **Magnitude ($M_w$)** | 6.0 – 7.5 | Destructive events relevant for code design |
| **Distance ($R_{jb}$)** | 10 – 50 km | Near-to-moderate field |
| **Site class ($V_{s30}$)** | 180 – 760 m/s | NEHRP C/D (stiff soil to soft rock) |
| **Fault mechanism** | Any | Captures variability |
| **Min records** | ≥ 200 | Statistical robustness for ML training |

### Step 3: Download

1. Select all records passing the filter (both horizontal components).
2. Choose **File Format → AT2 (unscaled)**.
3. Download the `.zip` archive.
4. Extract all `.AT2` files into **this directory** (`data/external/peer_nga/`).

### Step 4: (Optional) Download the Flatfile

Download `NGA_West2_flatfile.csv` from the same site and place it at:

```
data/external/NGA_West2_flatfile.csv
```

This enables automatic metadata filtering (Mw, Rjb, Vs30) by `data_factory.py`.

### Step 5: Run the Data Factory

```bash
# Dry run (parse + scale, no NLTHA):
python -m src.preprocessing.data_factory --dry-run

# Full campaign (parse + scale + NLTHA for all records):
python -m src.preprocessing.data_factory
```

## Expected File Format

PEER AT2 files follow this structure:

```
PEER NGA STRONG MOTION DATABASE RECORD
NORTHRIDGE 01/17/94 12:31, MUL CANYON, 009
NPTS=  3000, DT=    .0100 SEC
ACCELERATION IN UNITS OF G
  0.00583  0.00462  0.00354  0.00287  0.00264  ...
  ...
```

## Important Notes

- **Do NOT commit** AT2 files to Git (they are excluded by `.gitignore`).
- The `data_factory.py` script handles unit detection and conversion.
- Scale factors are computed automatically per ASCE 7-22 §16.2.
- Records exceeding the maximum scale factor (default: 5.0) are rejected.

## Reproducibility

For exact dataset replication, use the RSN list in `factory_summary.csv`
(generated after running the factory). Any reviewer can re-download the
identical records from PEER using those RSN numbers.
