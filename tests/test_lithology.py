import unittest

import pandas as pd

from lithology import (
    data_fingerprint,
    filter_bad_hole_rows,
    filter_selected_values,
    make_synthetic_data,
    predict_lithology,
    prepare_csv_download,
    train_models,
    validate_training_data,
)


class LithologyWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data = make_synthetic_data(rows_per_class=12, random_state=7)

    def test_training_uses_only_numeric_log_features(self) -> None:
        result = train_models(self.data, random_state=7)

        self.assertEqual(
            set(result.comparison["Model"]),
            {"Random Forest", "Support Vector Machine", "K-Nearest Neighbors"},
        )
        self.assertNotIn("Depth", result.feature_columns)
        self.assertNotIn("Wellbore Name", result.feature_columns)
        self.assertGreaterEqual(result.test_accuracy, 0.0)
        self.assertLessEqual(result.test_accuracy, 1.0)

    def test_prediction_adds_a_result_without_loading_external_models(self) -> None:
        result = train_models(self.data, random_state=7)
        prediction_input = self.data.drop(columns=["Lithology"])

        output = predict_lithology(prediction_input, result)

        self.assertEqual(len(output), len(prediction_input))
        self.assertTrue(output["Predicted Lithology"].notna().all())

    def test_prediction_rejects_missing_model_features(self) -> None:
        result = train_models(self.data, random_state=7)
        prediction_input = self.data.drop(
            columns=["Lithology", result.feature_columns[0]]
        )

        with self.assertRaisesRegex(ValueError, "missing columns"):
            predict_lithology(prediction_input, result)

    def test_training_rejects_unlabeled_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "Lithology"):
            validate_training_data(self.data.drop(columns=["Lithology"]))

    def test_bad_hole_filter_keeps_missing_measurements(self) -> None:
        frame = pd.DataFrame(
            {"CAL": [8.6, 10.0, None], "BS": [8.5, 8.5, 8.5], "value": [1, 2, 3]}
        )

        filtered = filter_bad_hole_rows(frame, tolerance=0.5)

        self.assertEqual(filtered["value"].tolist(), [1, 3])

    def test_minimum_two_rows_per_class_can_train(self) -> None:
        minimum = self.data.groupby("Lithology", group_keys=False).head(2)

        result = train_models(minimum, random_state=7)

        self.assertEqual(len(result.comparison), 3)

    def test_default_metadata_selection_preserves_null_rows(self) -> None:
        frame = pd.DataFrame({"Wellbore Name": ["Demo Well", None], "GR": [1, 2]})

        filtered = filter_selected_values(
            frame, "Wellbore Name", ["Demo Well"], ["Demo Well"]
        )

        self.assertEqual(len(filtered), 2)

    def test_data_fingerprint_changes_with_training_data(self) -> None:
        changed = self.data.copy()
        changed.loc[0, "GR"] += 1

        self.assertNotEqual(data_fingerprint(self.data), data_fingerprint(changed))

    def test_csv_download_neutralizes_formula_prefixes(self) -> None:
        frame = pd.DataFrame({"text": ["=1+1", " +SUM(A1:A2)", "@command", "ordinary"]})

        content = prepare_csv_download(frame).decode("utf-8")

        self.assertIn("'=1+1", content)
        self.assertIn("' +SUM(A1:A2)", content)
        self.assertIn("'@command", content)
        self.assertIn("ordinary", content)


if __name__ == "__main__":
    unittest.main()
