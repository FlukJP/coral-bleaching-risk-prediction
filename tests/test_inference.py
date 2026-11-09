import pandas as pd
import pytest

from model_inference import prepare_input_frame


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
