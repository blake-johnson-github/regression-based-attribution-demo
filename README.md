# Regression-Based Attribution (RBA) Demo

An end-to-end Python project demonstrating **regression-based attribution (RBA)** workflow.

Workflow includes fitting **ElasticNet** model with media response transformations (i.e., **ad stock** (lagged carryover) and **saturation** (diminishing returns)), ultimately producing output showing the contribution and ROI for each channel.

> **Note:** This repo was recreated to serve as a portfolio demo. All **client data, identifiers, and other potentially sensitive information** has been removed. Included data is **synthetic** and for demonstration purposes only.

## Quickstart

```bash
git clone <repo-url>
cd rba-analysis
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run pipeline with synthetic sample data:

```bash
rba-pipeline --config config/attribution.yml
```

Register Jupyter kernel for notebooks:

```bash
python -m ipykernel install --user --name attrib-reg --display-name "Python (attrib-reg)"
```

## Run pipeline

```bash
rba-pipeline --config config/attribution.yml
```

Running pipeline will read data, apply ad stock and saturation data transformations, fit time-series model (ElasticNet), and write results to `reports/`:

| Output File                    | Contents                                     |
|--------------------------------|----------------------------------------------|
| `config_used.yml`              | Copy of config                               |
| `coef_table.csv`               | Feature coefficients sorted by value         |
| `cv_metrics.csv`               | Per-fold cross-validation MAPE and R²        |
| `contributions_timeseries.csv` | Daily contribution by feature                |
| `contribution_totals.csv`      | Total contribution by feature                |
| `roi_summary.csv`              | ROI per media channel (contribution / spend) |

## Project structure

```
├── config/
│   ├── attribution.yml        # Default pipeline config
│   └── sample.yml             # Smaller grid for quick testing
├── data/
│   └── sample/                # Synthetic sample data
├── notebooks/                 # Data analysis
├── src/attrib_regression/     # Installable package
│   ├── cli.py                 # Pipeline entry point
│   ├── config.py              # YAML loading + namespace wrapper
│   ├── validation.py          # Input data validation
│   ├── io.py                  # CSV/Parquet/Excel reader
│   ├── preprocess.py          # Date parsing, sorting
│   ├── features/              # Ad stock, saturation, design matrix
│   ├── models/                # ElasticNet training + diagnostics
│   ├── attribution/           # Contribution decomposition + ROI
│   ├── eval/                  # Time-series cross-validation
│   └── viz/                   # Plotting utilities
├── tests/                     # Unit tests
├── outputs/                   # Model artifacts
└── reports/                   # Output tables
```

## Configuration

All pipeline parameters set in `config/attribution.yml`:

- **data** - input file path, date column, conversion (kpi/target) column
- **variables** - media spend columns, control columns
- **transforms.adstock** - channel decay and max lag
- **transforms.saturation** - Hill function/channel slope
- **model** - Hyperparameter grid, CV splits, positive constraint, standardization
- **outputs** - Models, figures, reports

## Data

A synthetic sample dataset (`data/sample/sample_daily.csv`) is included with 90 days of daily data:

- `date` - observation date
- `total_conversions` - target KPI
- `direct_spend`, `online_vid_spend`, `prog_ctv_spend`, `prog_display_spend`, `social_spend`, `stream_audio_spend` - media spend by channel

## Notebooks

Sequential workflow in `notebooks/`:

1. `01_data_audit.ipynb` - Data exploration and validation
2. `02_feature_engineering.ipynb` - Ad stock/saturation exploration
3. `03_model_fit.ipynb` - Model training and diagnostics
4. `04_attribution_outputs.ipynb` - Contribution decomposition and ROI
5. `05_sensitivity.ipynb` - Parameter sensitivity analysis
