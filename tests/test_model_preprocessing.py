import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from model_inference import FEATURE_COLUMNS as INFERENCE_FEATURE_COLUMNS
from model_preprocessing import (
    BASELINE_REALM_KEY,
    BASELINE_REALM_NAME,
    FEATURE_COLUMNS,
    FEATURE_SCHEMA,
    NON_CLIPPABLE_COLUMNS,
    REALM_INDICATOR_COLUMNS,
    ModelPreprocessor,
    select_features,
)


ROOT = Path(__file__).resolve().parents[1]


def make_features(rows=200):
    values = {}
    for column in FEATURE_COLUMNS:
        if column in REALM_INDICATOR_COLUMNS or column.startswith(
            ("Ocean_Name_", "Exposure_", "Substrate_Name_")
        ):
            values[column] = np.zeros(rows, dtype=bool)
        elif column == "Is_Tropical":
            values[column] = np.zeros(rows, dtype=int)
        elif column == "Aux_Marine_Heatwave":
            values[column] = np.arange(rows) % 2
        elif column == "Date_Day":
            values[column] = np.ones(rows, dtype=int)
        elif column == "Date_Month":
            values[column] = np.ones(rows, dtype=int)
        elif column == "Date_Year":
            values[column] = np.full(rows, 2000, dtype=int)
        elif column in {"Latitude_Degrees", "Longitude_Degrees"}:
            values[column] = np.arange(rows, dtype=float)
        else:
            values[column] = np.arange(rows, dtype=float) + 10
    return pd.DataFrame(values, columns=FEATURE_COLUMNS)


def test_coordinate_date_and_binary_columns_are_not_clipped():
    training = make_features()
    training.loc[0, "Latitude_Degrees"] = -999
    training.loc[1, "Longitude_Degrees"] = 999
    training.loc[2, "Date_Year"] = 9999
    training.loc[3, "Date_Month"] = 13
    training.loc[4, "Date_Day"] = 32

    preprocessor = ModelPreprocessor().fit(training)
    transformed = preprocessor.transform(training)

    for column in (
        "Latitude_Degrees",
        "Longitude_Degrees",
        "Date_Year",
        "Date_Month",
        "Date_Day",
        "Is_Tropical",
        "Aux_Marine_Heatwave",
    ):
        pd.testing.assert_series_equal(
            transformed[column], training[column].astype(float), check_names=True
        )
    assert "Percent_Bleaching" in NON_CLIPPABLE_COLUMNS
    assert transformed.loc[0, "Turbidity"] == preprocessor.clip_bounds["Turbidity"][
        "lower"
    ]


def test_validation_rows_do_not_change_train_fitted_bounds_or_medians():
    training = make_features(100)
    preprocessor = ModelPreprocessor().fit(training)
    state_before = preprocessor.to_state()

    validation = make_features(2)
    validation.loc[:, "Turbidity"] = [1_000_000, -1_000_000]
    validation.loc[:, "Aux_pH_Level"] = [500, -500]
    transformed = preprocessor.transform(validation)

    assert preprocessor.to_state() == state_before
    assert transformed["Turbidity"].max() == state_before["clip_bounds"][
        "Turbidity"
    ]["upper"]
    assert transformed["Turbidity"].min() == state_before["clip_bounds"][
        "Turbidity"
    ]["lower"]


def test_realm_medians_fall_back_to_training_global_for_unseen_realm():
    training = make_features(4)
    training.loc[:, "Turbidity"] = [10, 20, np.nan, np.nan]
    training.loc[2, REALM_INDICATOR_COLUMNS[0]] = True
    training.loc[3, REALM_INDICATOR_COLUMNS[0]] = True
    preprocessor = ModelPreprocessor().fit(training)

    validation = make_features(1)
    validation.loc[0, REALM_INDICATOR_COLUMNS[1]] = True
    validation.loc[0, "Turbidity"] = np.nan

    transformed = preprocessor.transform(validation)

    assert REALM_INDICATOR_COLUMNS[1] not in preprocessor.realm_medians
    assert transformed.loc[0, "Turbidity"] == preprocessor.global_medians["Turbidity"]


def test_feature_allowlist_ignores_ids_and_new_indicator_columns():
    data = make_features(2)
    data["Site_ID"] = ["site-1", "site-2"]
    data["Sample_ID"] = ["sample-1", "sample-2"]
    data["Matched_Indicator"] = [True, False]

    selected = select_features(data)

    assert tuple(selected.columns) == FEATURE_COLUMNS
    assert not {"Site_ID", "Sample_ID", "Matched_Indicator"}.intersection(
        selected.columns
    )
    assert INFERENCE_FEATURE_COLUMNS == FEATURE_COLUMNS
    boolean_features = {
        column for column, kind in FEATURE_SCHEMA.items() if kind == "bool"
    }
    assert boolean_features.issubset(NON_CLIPPABLE_COLUMNS)


def test_allowlist_matches_shipped_artifact_feature_order():
    stacking_artifact = joblib.load(
        ROOT / "model" / "stacking_ensemble" / "Stacking_Ensemble_model.joblib"
    )
    mlp_artifact = torch.load(
        ROOT / "model" / "custom_residual_mlp" / "custom_residual_mlp_artifact.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert tuple(stacking_artifact["original_feature_names"]) == FEATURE_COLUMNS
    assert tuple(mlp_artifact["config"]["feature_names"]) == FEATURE_COLUMNS


def test_artifact_state_round_trip_and_baseline_realm_contract():
    data = make_features()
    preprocessor = ModelPreprocessor().fit(data)
    state = preprocessor.to_state()
    restored = ModelPreprocessor.from_state(state)

    transformed = restored.transform(data.iloc[:3])
    assert tuple(transformed.columns) == FEATURE_COLUMNS
    assert BASELINE_REALM_KEY == "__baseline__"
    assert BASELINE_REALM_NAME == "Central Indo-Pacific"
    source = pd.read_csv(
        ROOT / "data" / "global_bleaching_environmental.csv",
        usecols=["Percent_Bleaching", "Realm_Name"],
        low_memory=False,
    )
    target = pd.to_numeric(
        source["Percent_Bleaching"].replace(
            ["nd", "ND", "na", "NA", "None", "none", ""], np.nan
        ),
        errors="coerce",
    )
    realms = sorted(source.loc[target.notna(), "Realm_Name"].dropna().unique())
    assert realms[0] == BASELINE_REALM_NAME
    model_ready = pd.read_csv(ROOT / "data" / "bleaching_model_ready.csv")
    assert model_ready[list(REALM_INDICATOR_COLUMNS)].sum(axis=1).eq(0).any()
    assert restored.realm_medians[BASELINE_REALM_KEY]["Turbidity"] == pytest.approx(
        data["Turbidity"].median()
    )


def test_artifact_state_rejects_missing_or_malformed_fields():
    state = ModelPreprocessor().fit(make_features()).to_state()
    missing_bounds = copy.deepcopy(state)
    del missing_bounds["clip_bounds"]["Turbidity"]
    with pytest.raises(ValueError, match="clip_bounds do not match"):
        ModelPreprocessor.from_state(missing_bounds)

    bad_version = copy.deepcopy(state)
    bad_version["schema_version"] = 2
    with pytest.raises(ValueError, match="unsupported or missing schema_version"):
        ModelPreprocessor.from_state(bad_version)

    bad_median = copy.deepcopy(state)
    bad_median["imputation"]["global_medians"]["Turbidity"] = "not-a-number"
    with pytest.raises(ValueError, match="global_medians.Turbidity must be numeric"):
        ModelPreprocessor.from_state(bad_median)


def test_realm_indicator_validation_is_explicit():
    training = make_features(3)
    training.loc[1, REALM_INDICATOR_COLUMNS[0]] = True
    training.loc[1, REALM_INDICATOR_COLUMNS[1]] = True
    with pytest.raises(ValueError, match="multiple active Realm indicators"):
        ModelPreprocessor().fit(training)
