"""Core data preparation and modeling for the lithology demo."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

TARGET_COLUMN = "Lithology"
METADATA_COLUMNS = {
    "Depth",
    "Well",
    "Wellbore Name",
    "TOPS",
    TARGET_COLUMN,
}


@dataclass(frozen=True)
class TrainingResult:
    """A fitted model and the evidence used to select it."""

    comparison: pd.DataFrame
    model: Pipeline
    feature_columns: tuple[str, ...]
    test_accuracy: float
    training_fingerprint: str


def data_fingerprint(frame: pd.DataFrame) -> str:
    """Return a stable fingerprint for invalidating models after data changes."""
    digest = hashlib.sha256()
    digest.update(repr(tuple(zip(frame.columns, frame.dtypes.astype(str)))).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def filter_selected_values(
    frame: pd.DataFrame,
    column: str,
    selected_values: list[str],
    available_values: list[str],
) -> pd.DataFrame:
    """Filter a metadata column while preserving nulls when all values are selected."""
    if set(selected_values) == set(available_values):
        return frame.copy()
    return frame[frame[column].astype(str).isin(selected_values)].copy()


def validate_training_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return numeric model inputs and labels after validating the public contract."""
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Training data must include a '{TARGET_COLUMN}' column.")

    labeled = frame.dropna(subset=[TARGET_COLUMN]).copy()
    if labeled.empty:
        raise ValueError("Training data contains no labeled rows.")

    labels = labeled[TARGET_COLUMN].astype(str)
    class_counts = labels.value_counts()
    if len(class_counts) < 2:
        raise ValueError("Training data must contain at least two lithology classes.")
    if class_counts.min() < 2:
        raise ValueError("Each lithology class needs at least two labeled rows.")

    candidate_columns = [
        column for column in labeled.columns if column not in METADATA_COLUMNS
    ]
    features = labeled[candidate_columns].apply(pd.to_numeric, errors="coerce")
    features = features.dropna(axis=1, how="all")
    if features.empty:
        raise ValueError("Training data must contain at least one numeric log column.")

    return features, labels


def train_models(frame: pd.DataFrame, random_state: int = 42) -> TrainingResult:
    """Compare three classifiers and return the best fitted pipeline."""
    features, labels = validate_training_data(frame)
    class_count = labels.nunique()
    test_rows = max(class_count, math.ceil(len(labels) * 0.25))
    test_rows = min(test_rows, len(labels) - class_count)
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=test_rows,
        random_state=random_state,
        stratify=labels,
    )

    neighbor_count = max(1, min(7, len(train_x) - 1))
    models: dict[str, Pipeline] = {
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "Support Vector Machine": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("classifier", SVC()),
            ]
        ),
        "K-Nearest Neighbors": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=neighbor_count)),
            ]
        ),
    }

    rows: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    for name, model in models.items():
        model.fit(train_x, train_y)
        fitted[name] = model
        rows.append(
            {
                "Model": name,
                "Training accuracy": accuracy_score(train_y, model.predict(train_x)),
                "Test accuracy": accuracy_score(test_y, model.predict(test_x)),
            }
        )

    comparison = pd.DataFrame(rows).sort_values(
        ["Test accuracy", "Training accuracy"], ascending=False
    )
    winner = str(comparison.iloc[0]["Model"])
    return TrainingResult(
        comparison=comparison.reset_index(drop=True),
        model=fitted[winner],
        feature_columns=tuple(features.columns),
        test_accuracy=float(comparison.iloc[0]["Test accuracy"]),
        training_fingerprint=data_fingerprint(frame),
    )


def predict_lithology(frame: pd.DataFrame, result: TrainingResult) -> pd.DataFrame:
    """Apply a fitted pipeline to data that contains the same numeric features."""
    missing = [
        column for column in result.feature_columns if column not in frame.columns
    ]
    if missing:
        raise ValueError("Prediction data is missing columns: " + ", ".join(missing))

    features = frame[list(result.feature_columns)].apply(pd.to_numeric, errors="coerce")
    output = frame.copy()
    output["Predicted Lithology"] = result.model.predict(features)
    return output


def filter_bad_hole_rows(frame: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Remove rows whose absolute caliper and bit-size difference exceeds tolerance."""
    if "CAL" not in frame.columns or "BS" not in frame.columns:
        return frame.copy()

    caliper = pd.to_numeric(frame["CAL"], errors="coerce")
    bit_size = pd.to_numeric(frame["BS"], errors="coerce")
    difference = (caliper - bit_size).abs()
    keep = difference.isna() | difference.le(tolerance)
    return frame.loc[keep].copy()


def prepare_csv_download(frame: pd.DataFrame) -> bytes:
    """Serialize CSV while neutralizing spreadsheet formula prefixes in text."""
    safe = frame.copy()

    def escape_formula(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        candidate = value.lstrip(" \t\r\n")
        if candidate.startswith(("=", "+", "-", "@")):
            return "'" + value
        return value

    for column in safe.select_dtypes(include=["object", "string"]).columns:
        safe[column] = safe[column].map(escape_formula)
    return safe.to_csv(index=False).encode("utf-8")


def make_synthetic_data(
    rows_per_class: int = 50, random_state: int = 42
) -> pd.DataFrame:
    """Create fictional data for exercising the workflow without restricted records."""
    if rows_per_class < 4:
        raise ValueError("rows_per_class must be at least 4.")

    rng = np.random.default_rng(random_state)
    profiles = {
        "Claystone": (88.0, 5.5, 2.45, 0.34),
        "Sandstone": (42.0, 15.0, 2.18, 0.22),
        "Limestone": (24.0, 4.0, 2.68, 0.12),
        "Marl": (66.0, 7.5, 2.52, 0.28),
    }
    records: list[dict[str, Any]] = []
    for class_index, (lithology, profile) in enumerate(profiles.items()):
        gamma_ray, penetration_rate, density, neutron = profile
        for row_index in range(rows_per_class):
            records.append(
                {
                    "Wellbore Name": f"Demo Well {(row_index % 2) + 1}",
                    "TOPS": f"Formation {(class_index % 2) + 1}",
                    "Depth": 1000.0 + class_index * 250 + row_index * 0.5,
                    "ROP": rng.normal(penetration_rate, 1.2),
                    "GR": rng.normal(gamma_ray, 4.0),
                    "RES_attenuation": rng.lognormal(0.1 + class_index * 0.2, 0.12),
                    "RES_phase": rng.lognormal(0.2 + class_index * 0.15, 0.1),
                    "CAL": rng.normal(8.55, 0.12),
                    "BS": 8.5,
                    "DEN": rng.normal(density, 0.04),
                    "NEU": rng.normal(neutron, 0.025),
                    "DTC": rng.normal(80 - class_index * 7, 2.5),
                    TARGET_COLUMN: lithology,
                }
            )
    return pd.DataFrame.from_records(records)
