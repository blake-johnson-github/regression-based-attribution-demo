# Regression-Based Attribution (RBA) Demo

An end-to-end Python pipeline for regression-based attribution in marketing analytics. This project uses ElasticNet regression with adstock and saturation transformations to estimate channel-level media performance contribution and ROI.

> **Portfolio Note:** This project was recreated as a portfolio demonstration. Data provided is synthetic and all client data, identifiers, and/or other potentially sensitive information has been removed.

## Contents

- Config-driven CLI pipeline (`rba-pipeline`)
- Media response transformations (adstock + saturation)
- ElasticNet training with time-series cross-validation
- Contribution decomposition and ROI reporting
- Sequential Jupyter notebooks walking through the analysis
- Unit tests and development tooling (pytest, ruff, black)

## Quick Start

```bash
git clone https://github.com/blake-johnson-github/regression-based-attribution-demo.git
cd regression-based-attribution-demo

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run the pipeline on the included sample data:

```bash
rba-pipeline --config config/attribution.yml
```

To use the notebooks, register a Jupyter kernel:

```bash
python -m ipykernel install --user --name attrib-reg --display-name "Python (attrib-reg)"
```

## Pipeline Output

The pipeline writes results to `reports/`:

| File                           | Description                                  |
|--------------------------------|----------------------------------------------|
| `config_used.yml`              | Configuration snapshot for reproducibility   |
| `coef_table.csv`               | Feature coefficients sorted by magnitude     |
| `cv_metrics.csv`               | Cross-validation metrics (MAPE and R²)       |
| `contributions_timeseries.csv` | Daily contribution by feature                |
| `contribution_totals.csv`      | Total contribution by feature                |
| `roi_summary.csv`              | ROI per media channel (contribution / spend) |

## Project Structure

```
├── config/
│   ├── attribution.yml        # Default pipeline configuration
│   └── sample.yml             # Smaller grid for testing
├── data/
│   └── sample/                # Synthetic sample data (90 days)
├── notebooks/                 # Sequential analysis workflow
├── src/attrib_regression/     # Main package
│   ├── cli.py                 # Pipeline entry point
│   ├── config.py              # Configuration loading
│   ├── validation.py          # Input data validation
│   ├── io.py                  # Data readers (CSV/Parquet/Excel)
│   ├── preprocess.py          # Date parsing and sorting
│   ├── features/              # Adstock, saturation, design matrix
│   ├── models/                # ElasticNet training and diagnostics
│   ├── attribution/           # Contribution decomposition and ROI
│   ├── eval/                  # Time-series cross-validation
│   └── viz/                   # Plotting utilities
├── tests/                     # Unit tests
├── outputs/                   # Model artifacts (gitignored)
└── reports/                   # Pipeline results (gitignored)
```

## Configuration

Parameters are defined in `config/attribution.yml`:

- **data** - Input file path, date column, target (KPI) column
- **variables** - Media spend columns and control variables
- **transforms.adstock** - Per-channel decay rates and maximum lag
- **transforms.saturation** - Hill function parameters (ec50/slope)
- **model** - ElasticNet hyperparameter grid, CV splits, constraints
- **outputs** - Directories for models, figures, and reports

## Sample Data

The synthetic dataset (`data/sample/sample_daily.csv`) includes 90 days of daily observations:

- `date` - Observation date
- `total_conversions` - Target KPI
- `direct_spend`, `online_vid_spend`, `prog_ctv_spend`, `prog_display_spend`, `social_spend`, `stream_audio_spend` - Media spend by channel

## Notebooks

The analysis workflow is organized into five sequential notebooks:

1. `01_data_audit.ipynb` - Data exploration and validation
2. `02_feature_engineering.ipynb` - Adstock and saturation exploration
3. `03_model_fit.ipynb` - Model training and diagnostics
4. `04_attribution_outputs.ipynb` - Contribution decomposition and ROI
5. `05_sensitivity.ipynb` - Parameter sensitivity analysis

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## Development

```bash
black src/
ruff check src/
nbqa ruff notebooks/
```

## About

This repo is a recreation of previous work I have done and is intended solely for portfolio purposes.

**Contact**
- Author: **Blake Johnson**
- LinkedIn: [linkedin.com/in/mblakejohnson](https://linkedin.com/in/mblakejohnson)
- GitHub: [@blake-johnson-github](https://github.com/blake-johnson-github)

## License

MIT License - Copyright (c) 2026 Blake Johnson
