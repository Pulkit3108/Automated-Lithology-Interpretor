"""Streamlit interface for the public lithology classification demo."""

from __future__ import annotations

from io import BytesIO
from zipfile import BadZipFile

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from openpyxl.utils.exceptions import InvalidFileException
from pandas.errors import ParserError
from sklearn.metrics import accuracy_score

from lithology import (
    TARGET_COLUMN,
    TrainingResult,
    data_fingerprint,
    filter_bad_hole_rows,
    filter_selected_values,
    make_synthetic_data,
    predict_lithology,
    prepare_csv_download,
    train_models,
)

st.set_page_config(page_title="Automated Lithology Interpreter", layout="wide")


@st.cache_data
def read_uploaded_file(name: str, content: bytes) -> pd.DataFrame:
    """Read a user-provided CSV or Excel workbook without external calls."""
    buffer = BytesIO(content)
    if name.lower().endswith(".csv"):
        return pd.read_csv(buffer)
    return pd.read_excel(buffer)


def select_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply optional well, formation, and bad-hole filters."""
    selected = frame.copy()
    if "Wellbore Name" in selected.columns:
        wells = sorted(selected["Wellbore Name"].dropna().astype(str).unique())
        chosen_wells = st.sidebar.multiselect("Wells", wells, default=wells)
        selected = filter_selected_values(
            selected, "Wellbore Name", chosen_wells, wells
        )
    if "TOPS" in selected.columns:
        formations = sorted(selected["TOPS"].dropna().astype(str).unique())
        chosen_formations = st.sidebar.multiselect(
            "Formations", formations, default=formations
        )
        selected = filter_selected_values(
            selected, "TOPS", chosen_formations, formations
        )
    if {"CAL", "BS"}.issubset(selected.columns):
        apply_filter = st.sidebar.checkbox("Apply caliper/bit-size filter")
        if apply_filter:
            tolerance = st.sidebar.number_input(
                "Maximum absolute difference", min_value=0.0, value=0.5, step=0.1
            )
            selected = filter_bad_hole_rows(selected, tolerance)
    return selected


st.title("Automated Lithology Interpreter")
st.write(
    "A local demonstration of tabular well-log exploration and supervised "
    "lithology classification. The bundled data is fictional and exists only "
    "to exercise the workflow."
)

source = st.radio(
    "Data source",
    ["Synthetic example", "Upload CSV or Excel"],
    horizontal=True,
)
if source == "Synthetic example":
    data = make_synthetic_data()
else:
    upload = st.file_uploader("Training data", type=["csv", "xlsx", "xlsm"])
    if upload is None:
        st.info("Upload a file to continue.")
        st.stop()
    try:
        data = read_uploaded_file(upload.name, upload.getvalue())
    except (
        BadZipFile,
        InvalidFileException,
        OSError,
        ParserError,
        UnicodeError,
        ValueError,
    ) as exc:
        st.error(f"The file could not be read: {exc}")
        st.stop()

filtered = select_rows(data)
if filtered.empty:
    st.warning("The selected filters produced no rows.")
    st.stop()

existing_result: TrainingResult | None = st.session_state.get("training_result")
if (
    existing_result is not None
    and existing_result.training_fingerprint != data_fingerprint(filtered)
):
    del st.session_state.training_result
    st.info("Training data changed. Train the models again before predicting.")

overview_tab, explore_tab, train_tab, predict_tab = st.tabs(
    ["Overview", "Explore", "Train", "Predict"]
)

with overview_tab:
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(filtered):,}")
    col2.metric("Columns", len(filtered.columns))
    class_count = filtered[TARGET_COLUMN].nunique() if TARGET_COLUMN in filtered else 0
    col3.metric("Lithology classes", class_count)
    st.dataframe(filtered.head(100), width="stretch")
    missing = filtered.isna().mean().mul(100).sort_values(ascending=False)
    st.subheader("Missing values")
    st.bar_chart(missing.rename("Missing percent"))

with explore_tab:
    numeric_columns = filtered.select_dtypes(include="number").columns.tolist()
    default_tracks = [
        column for column in ["GR", "ROP", "DEN", "NEU"] if column in numeric_columns
    ]
    tracks = st.multiselect("Log curves", numeric_columns, default=default_tracks)
    if tracks:
        depth = filtered.get("Depth", filtered.index)
        figure, axes = plt.subplots(1, len(tracks), figsize=(3 * len(tracks), 8))
        axes = [axes] if len(tracks) == 1 else axes
        for axis, column in zip(axes, tracks):
            axis.plot(filtered[column], depth)
            axis.set_title(column)
            axis.grid(alpha=0.25)
            axis.invert_yaxis()
        axes[0].set_ylabel("Depth" if "Depth" in filtered else "Row")
        figure.tight_layout()
        st.pyplot(figure)

    if TARGET_COLUMN in filtered.columns and numeric_columns:
        x_axis = st.selectbox("Distribution column", numeric_columns)
        st.dataframe(
            filtered.groupby(TARGET_COLUMN)[x_axis].describe().round(3),
            width="stretch",
        )

with train_tab:
    st.write(
        "Training uses a stratified holdout split. Preprocessing is fitted only "
        "on the training partition through each scikit-learn pipeline."
    )
    if st.button("Train and compare models", type="primary"):
        try:
            st.session_state.training_result = train_models(filtered)
        except ValueError as exc:
            st.error(str(exc))
    result: TrainingResult | None = st.session_state.get("training_result")
    if result is not None:
        comparison = result.comparison.copy()
        comparison["Training accuracy"] = comparison["Training accuracy"].map(
            lambda value: f"{value:.3f}"
        )
        comparison["Test accuracy"] = comparison["Test accuracy"].map(
            lambda value: f"{value:.3f}"
        )
        st.dataframe(comparison, hide_index=True, width="stretch")
        st.success(f"Selected model: {result.comparison.iloc[0]['Model']}")

with predict_tab:
    result = st.session_state.get("training_result")
    if result is None:
        st.info("Train the models before running predictions.")
    else:
        prediction_upload = st.file_uploader(
            "Prediction data", type=["csv", "xlsx", "xlsm"], key="prediction"
        )
        try:
            prediction_data = (
                filtered
                if prediction_upload is None
                else read_uploaded_file(
                    prediction_upload.name, prediction_upload.getvalue()
                )
            )
            predictions = predict_lithology(prediction_data, result)
            if TARGET_COLUMN in predictions.columns:
                labeled = predictions.dropna(subset=[TARGET_COLUMN])
                if not labeled.empty:
                    score = accuracy_score(
                        labeled[TARGET_COLUMN].astype(str),
                        labeled["Predicted Lithology"],
                    )
                    st.metric("Accuracy on labeled prediction rows", f"{score:.3f}")
            st.dataframe(predictions.head(200), width="stretch")
            st.download_button(
                "Download predictions",
                prepare_csv_download(predictions),
                file_name="lithology_predictions.csv",
                mime="text/csv",
            )
        except (
            BadZipFile,
            InvalidFileException,
            OSError,
            ParserError,
            UnicodeError,
            ValueError,
        ) as exc:
            st.error(f"Prediction data could not be processed: {exc}")
