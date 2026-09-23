# Automated Lithology Interpreter

Automated Lithology Interpreter is a cleaned, local demonstration of a hackathon idea: use tabular well-log measurements to explore formations, compare classification models, and predict lithology labels.

This public version contains no corporate integration, private endpoint, credential, restricted presentation, packaged model, or real well dataset. It makes no external network calls. The bundled example data is generated and fictional.

## Features

- Load a CSV or Excel workbook, or start with deterministic synthetic data.
- Filter records by well and formation when those fields are present.
- Review missing measurements and optional caliper-versus-bit-size filtering.
- Plot selected log curves against depth.
- Compare Random Forest, Support Vector Machine, and K-Nearest Neighbors classifiers.
- Keep preprocessing and model fitting in one scikit-learn pipeline to prevent train/test leakage.
- Apply the selected model to another dataset and download predictions as CSV.
- Neutralize spreadsheet formula prefixes in downloaded CSV text fields.

## Project structure

```text
.
├── app.py                    # Streamlit interface
├── lithology.py              # Validation, synthetic data, modeling, prediction
├── tests/
│   └── test_lithology.py     # Focused workflow tests
├── docs/
│   └── project-overview.md   # Hackathon context and design summary
└── requirements.txt
```

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Data contract

Training data must include:

- `Lithology`: the target class.
- At least one numeric measurement column.
- At least two lithology classes, with two or more rows per class.

The application treats `Depth`, `Well`, `Wellbore Name`, and `TOPS` as metadata rather than model features. Prediction data must contain the same numeric feature columns selected during training.

Common optional well-log columns include `ROP`, `GR`, `RES_attenuation`, `RES_phase`, `CAL`, `BS`, `DEN`, `NEU`, and `DTC`. Column meanings and units depend on the source dataset and must be documented by its owner.

## Safety and limitations

- Use only data that you are allowed to process and publish.
- The synthetic dataset is intended for workflow testing, not geological validation.
- Accuracy on one holdout split does not establish field performance.
- The application does not persist or deserialize model files.
- The application does not connect to live drilling or well-data systems.
- Predictions are educational and must not be used for drilling, safety, or operational decisions.

See [Project overview](docs/project-overview.md) for the original use case and the boundary of this public reconstruction.
