import numpy as np
import pandas as pd
import pytest
import torch

from model_inference import predict_all_models, prepare_input_frame
from model_preprocessing import FEATURE_COLUMNS, REALM_INDICATOR_COLUMNS, ModelPreprocessor


@pytest.fixture
def reference_and_artifacts():
    reference = pd.DataFrame(
        {
            "numeric": [1.0, 3.0],
            "flag": [False, True],
            "other": [2.0, 4.0],
        }
    )
    feature_names = ["numeric", "flag", "other"]
    stacking_artifact = {
        "original_feature_names": feature_names,
        "feature_names": feature_names,
    }
    mlp_artifact = {"config": {"feature_names": feature_names}}
    return reference, stacking_artifact, mlp_artifact


def test_prepare_input_frame_accepts_valid_input(reference_and_artifacts):
    reference, stacking_artifact, mlp_artifact = reference_and_artifacts
    input_frame = pd.DataFrame(
        {"numeric": ["2.5"], "flag": ["true"], "other": ["3.5"]}
    )

    prepared = prepare_input_frame(
        input_frame, reference, stacking_artifact, mlp_artifact
    )

    assert prepared.stacking.loc[0, "numeric"] == 2.5
    assert prepared.stacking.loc[0, "flag"] == 1
    assert prepared.mlp.loc[0, "other"] == 3.5
    assert prepared.fill_notes == []


def test_prepare_input_frame_fills_malformed_values(reference_and_artifacts):
    reference, stacking_artifact, mlp_artifact = reference_and_artifacts
    input_frame = pd.DataFrame(
        {"numeric": ["invalid"], "flag": [False], "other": ["5"]}
    )

    prepared = prepare_input_frame(
        input_frame, reference, stacking_artifact, mlp_artifact
    )

    assert prepared.stacking.loc[0, "numeric"] == 2.0
    assert prepared.stacking.loc[0, "flag"] == 0
    assert len(prepared.fill_notes) == 1


def test_legacy_artifact_fills_nan_from_reference_without_preprocessing_state(
    reference_and_artifacts,
):
    reference, stacking_artifact, mlp_artifact = reference_and_artifacts
    missing_input = pd.DataFrame(
        {"numeric": [float("nan")], "flag": [False], "other": [3.5]}
    )
    filled_input = pd.DataFrame(
        {"numeric": [2.0], "flag": [False], "other": [3.5]}
    )

    missing_result = prepare_input_frame(
        missing_input, reference, stacking_artifact, mlp_artifact
    )
    filled_result = prepare_input_frame(
        filled_input, reference, stacking_artifact, mlp_artifact
    )

    pd.testing.assert_frame_equal(missing_result.stacking, filled_result.stacking)
    pd.testing.assert_frame_equal(missing_result.mlp, filled_result.mlp)
    assert missing_result.stacking.loc[0, "numeric"] == 2.0
    assert missing_result.fill_notes == [
        "numeric: filled 1 missing numeric value(s) with 2.0000"
    ]

    class FixedModel:
        def predict(self, frame):
            return np.full(len(frame), np.log1p(4.0))

        def predict_proba(self, frame):
            return np.tile([0.0, 1.0], (len(frame), 1))

    class FixedScaler:
        def transform(self, frame):
            return frame.to_numpy(dtype=float)

    class FixedMLP:
        def __call__(self, tensor):
            return torch.full((len(tensor), 1), np.log1p(3.0))

    stacking_artifact.update(
        {
            "stacking_model": FixedModel(),
            "two_stage_classifier": FixedModel(),
            "two_stage_regressor": FixedModel(),
            "two_stage_threshold": 0.5,
            "blend_alpha": 0.5,
        }
    )
    mlp_artifact["scaler"] = FixedScaler()
    missing_predictions, _ = predict_all_models(
        missing_input,
        reference,
        stacking_artifact,
        mlp_artifact,
        FixedMLP(),
    )
    filled_predictions, _ = predict_all_models(
        filled_input,
        reference,
        stacking_artifact,
        mlp_artifact,
        FixedMLP(),
    )
    pd.testing.assert_frame_equal(missing_predictions, filled_predictions)


def _full_schema_frames():
    rows = 4
    values = {}
    for column in FEATURE_COLUMNS:
        if column in REALM_INDICATOR_COLUMNS or column.startswith(
            ("Ocean_Name_", "Exposure_", "Substrate_Name_")
        ):
            values[column] = [False] * rows
        elif column == "Is_Tropical":
            values[column] = [0] * rows
        elif column == "Aux_Marine_Heatwave":
            values[column] = [0.0, 1.0, 0.0, 1.0]
        else:
            values[column] = [1.0, 2.0, 3.0, 4.0]
    reference = pd.DataFrame(values, columns=FEATURE_COLUMNS)
    return reference


def test_v2_artifact_preprocessing_uses_serialized_imputation_values():
    reference = _full_schema_frames()
    preprocessor = ModelPreprocessor().fit(reference)
    state = preprocessor.to_state()
    state["imputation"]["global_medians"]["Turbidity"] = 100.0
    state["imputation"]["realm_medians"]["__baseline__"]["Turbidity"] = 42.0

    input_frame = reference.iloc[[0]].copy()
    input_frame.loc[:, "Turbidity"] = float("nan")
    stacking_artifact = {
        "artifact_version": 2,
        "original_feature_names": list(FEATURE_COLUMNS),
        "feature_names": list(FEATURE_COLUMNS),
        "preprocessing_state": state,
    }
    mlp_artifact = {"config": {"feature_names": list(FEATURE_COLUMNS)}}

    prepared = prepare_input_frame(
        input_frame, reference, stacking_artifact, mlp_artifact
    )

    assert prepared.stacking.loc[0, "Turbidity"] == 42.0
    assert prepared.mlp.loc[0, "Turbidity"] == reference["Turbidity"].median()


def test_v2_mlp_artifact_uses_its_own_preprocessing_state():
    reference = _full_schema_frames()
    state = ModelPreprocessor().fit(reference).to_state()
    state["imputation"]["global_medians"]["Turbidity"] = 100.0
    state["imputation"]["realm_medians"]["__baseline__"]["Turbidity"] = 84.0
    input_frame = reference.iloc[[0]].copy()
    input_frame.loc[:, "Turbidity"] = float("nan")
    stacking_artifact = {
        "artifact_version": 1,
        "original_feature_names": list(FEATURE_COLUMNS),
        "feature_names": list(FEATURE_COLUMNS),
    }
    mlp_artifact = {
        "artifact_version": 2,
        "config": {"feature_names": list(FEATURE_COLUMNS)},
        "preprocessing_state": state,
    }

    prepared = prepare_input_frame(
        input_frame, reference, stacking_artifact, mlp_artifact
    )

    assert prepared.stacking.loc[0, "Turbidity"] == reference["Turbidity"].median()
    assert prepared.mlp.loc[0, "Turbidity"] == 84.0


def test_v2_artifact_reports_malformed_preprocessing_state():
    reference = _full_schema_frames()
    state = ModelPreprocessor().fit(reference).to_state()
    state["schema_version"] = 999
    stacking_artifact = {
        "artifact_version": 2,
        "original_feature_names": list(FEATURE_COLUMNS),
        "feature_names": list(FEATURE_COLUMNS),
        "preprocessing_state": state,
    }
    mlp_artifact = {"config": {"feature_names": list(FEATURE_COLUMNS)}}

    with pytest.raises(
        ValueError, match="Invalid preprocessing_state: unsupported or missing schema_version."
    ):
        prepare_input_frame(
            reference.iloc[[0]], reference, stacking_artifact, mlp_artifact
        )


def test_v2_artifact_requires_preprocessing_state():
    reference = _full_schema_frames()
    stacking_artifact = {
        "artifact_version": 2,
        "original_feature_names": list(FEATURE_COLUMNS),
        "feature_names": list(FEATURE_COLUMNS),
    }
    mlp_artifact = {"config": {"feature_names": list(FEATURE_COLUMNS)}}

    with pytest.raises(
        ValueError, match="Invalid preprocessing_state: missing for artifact_version 2."
    ):
        prepare_input_frame(
            reference.iloc[[0]], reference, stacking_artifact, mlp_artifact
        )


def test_prepare_input_frame_reports_empty_input(reference_and_artifacts):
    reference, stacking_artifact, mlp_artifact = reference_and_artifacts

    with pytest.raises(ValueError, match="The input table is empty."):
        prepare_input_frame(
            pd.DataFrame(), reference, stacking_artifact, mlp_artifact
        )


def test_prepare_input_frame_reports_missing_required_columns(
    reference_and_artifacts,
):
    reference, stacking_artifact, mlp_artifact = reference_and_artifacts
    input_frame = pd.DataFrame({"numeric": [1.0], "flag": [True]})

    with pytest.raises(
        ValueError,
        match="Missing required feature columns.*other",
    ):
        prepare_input_frame(
            input_frame, reference, stacking_artifact, mlp_artifact
        )
