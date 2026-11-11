from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


TARGET_NAME = "Percent_Bleaching"

FEATURE_SCHEMA = {
    "Latitude_Degrees": "float",
    "Longitude_Degrees": "float",
    "Distance_to_Shore": "float",
    "Turbidity": "float",
    "Cyclone_Frequency": "float",
    "Date_Day": "integer",
    "Date_Month": "integer",
    "Date_Year": "integer",
    "Depth_m": "float",
    "ClimSST": "float",
    "Temperature_Mean": "float",
    "Temperature_Minimum": "float",
    "Temperature_Maximum": "float",
    "Temperature_Kelvin_Standard_Deviation": "float",
    "Windspeed": "float",
    "SSTA": "float",
    "SSTA_Standard_Deviation": "float",
    "SSTA_Mean": "float",
    "SSTA_Minimum": "float",
    "SSTA_Maximum": "float",
    "SSTA_Frequency": "float",
    "SSTA_Frequency_Standard_Deviation": "float",
    "SSTA_FrequencyMax": "float",
    "SSTA_FrequencyMean": "float",
    "SSTA_DHW": "float",
    "SSTA_DHW_Standard_Deviation": "float",
    "SSTA_DHWMax": "float",
    "SSTA_DHWMean": "float",
    "TSA": "float",
    "TSA_Standard_Deviation": "float",
    "TSA_Minimum": "float",
    "TSA_Maximum": "float",
    "TSA_Mean": "float",
    "TSA_Frequency": "float",
    "TSA_Frequency_Standard_Deviation": "float",
    "TSA_FrequencyMax": "float",
    "TSA_FrequencyMean": "float",
    "TSA_DHW": "float",
    "TSA_DHW_Standard_Deviation": "float",
    "TSA_DHWMax": "float",
    "TSA_DHWMean": "float",
    "Temperature_C": "float",
    "Temp_Range_C": "float",
    "Is_Tropical": "integer",
    "Aux_Marine_Heatwave": "float",
    "Aux_pH_Level": "float",
    "Ocean_Name_Atlantic": "bool",
    "Ocean_Name_Indian": "bool",
    "Ocean_Name_Pacific": "bool",
    "Ocean_Name_Red Sea": "bool",
    "Realm_Name_Eastern Indo-Pacific": "bool",
    "Realm_Name_Temperate Australasia": "bool",
    "Realm_Name_Temperate Northern Atlantic": "bool",
    "Realm_Name_Temperate Northern Pacific": "bool",
    "Realm_Name_Tropical Atlantic": "bool",
    "Realm_Name_Tropical Eastern Pacific": "bool",
    "Realm_Name_Western Indo-Pacific": "bool",
    "Exposure_Sheltered": "bool",
    "Exposure_Sometimes": "bool",
    "Substrate_Name_Hard Coral": "bool",
    "Substrate_Name_Nutrient Indicator Algae": "bool",
}

FEATURE_COLUMNS = tuple(FEATURE_SCHEMA)
TARGET_INSERT_INDEX = FEATURE_COLUMNS.index("ClimSST")
MODEL_READY_COLUMNS = (
    *FEATURE_COLUMNS[:TARGET_INSERT_INDEX],
    TARGET_NAME,
    *FEATURE_COLUMNS[TARGET_INSERT_INDEX:],
)

REALM_INDICATOR_COLUMNS = tuple(
    column for column in FEATURE_COLUMNS if column.startswith("Realm_Name_")
)
BASELINE_REALM_KEY = "__baseline__"
BASELINE_REALM_NAME = "Central Indo-Pacific"

NON_CLIPPABLE_COLUMNS = (
    "Latitude_Degrees",
    "Longitude_Degrees",
    "Date_Year",
    "Date_Month",
    "Date_Day",
    TARGET_NAME,
    "Is_Tropical",
    "Aux_Marine_Heatwave",
    "Ocean_Name_Atlantic",
    "Ocean_Name_Indian",
    "Ocean_Name_Pacific",
    "Ocean_Name_Red Sea",
    *REALM_INDICATOR_COLUMNS,
    "Exposure_Sheltered",
    "Exposure_Sometimes",
    "Substrate_Name_Hard Coral",
    "Substrate_Name_Nutrient Indicator Algae",
)

REQUIRED_NON_NULL_COLUMNS = (
    TARGET_NAME,
    "Latitude_Degrees",
    "Longitude_Degrees",
    "Date_Year",
    "Date_Month",
    "Date_Day",
    "Is_Tropical",
    "Ocean_Name_Atlantic",
    "Ocean_Name_Indian",
    "Ocean_Name_Pacific",
    "Ocean_Name_Red Sea",
    *REALM_INDICATOR_COLUMNS,
    "Exposure_Sheltered",
    "Exposure_Sometimes",
    "Substrate_Name_Hard Coral",
    "Substrate_Name_Nutrient Indicator Algae",
)

CLIP_LOWER_QUANTILE = 0.005
CLIP_UPPER_QUANTILE = 0.995
IMPUTATION_STRATEGY = "realm_median_then_global_median"
PREPROCESSING_SCHEMA_VERSION = 1
ARTIFACT_PREPROCESSING_FIT_SCOPE = "full_training_set"
EVALUATION_PREPROCESSING_FIT_SCOPE = "per_fold_training_rows"


def select_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")
    return frame.loc[:, FEATURE_COLUMNS].copy()


def _numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = select_features(frame).apply(pd.to_numeric, errors="coerce")
    numeric = numeric.astype(float)
    return numeric.replace([np.inf, -np.inf], np.nan)


def _realm_keys(frame: pd.DataFrame) -> list[str]:
    indicators = frame.loc[:, REALM_INDICATOR_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    values = indicators.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isin(values, [0.0, 1.0]).all():
        raise ValueError(
            "Realm indicator columns must contain only non-missing 0/1 values."
        )

    active_counts = values.sum(axis=1)
    if np.any(active_counts > 1):
        raise ValueError("A row cannot have multiple active Realm indicators.")

    keys = []
    for row in values:
        active = np.flatnonzero(row)
        keys.append(
            REALM_INDICATOR_COLUMNS[active[0]]
            if len(active)
            else BASELINE_REALM_KEY
        )
    return keys


class ModelPreprocessor:
    def __init__(
        self,
        lower_quantile: float = CLIP_LOWER_QUANTILE,
        upper_quantile: float = CLIP_UPPER_QUANTILE,
    ) -> None:
        if not 0 <= lower_quantile < upper_quantile <= 1:
            raise ValueError("Clip quantiles must satisfy 0 <= lower < upper <= 1.")
        self.lower_quantile = float(lower_quantile)
        self.upper_quantile = float(upper_quantile)
        self.clip_bounds: dict[str, dict[str, float]] = {}
        self.global_medians: dict[str, float] = {}
        self.realm_medians: dict[str, dict[str, float]] = {}
        self._is_fitted = False

    @property
    def clippable_columns(self) -> tuple[str, ...]:
        excluded = set(NON_CLIPPABLE_COLUMNS)
        return tuple(column for column in FEATURE_COLUMNS if column not in excluded)

    def fit(self, frame: pd.DataFrame) -> ModelPreprocessor:
        if frame.empty:
            raise ValueError("Cannot fit preprocessing state on an empty dataset.")
        values = _numeric_features(frame)
        realm_keys = _realm_keys(values)

        self.clip_bounds = {}
        for column in self.clippable_columns:
            non_missing = values[column].dropna()
            if non_missing.empty:
                raise ValueError(
                    f"Cannot fit clipping bounds: {column} has no finite training values."
                )
            self.clip_bounds[column] = {
                "lower": float(non_missing.quantile(self.lower_quantile)),
                "upper": float(non_missing.quantile(self.upper_quantile)),
            }

        clipped = self._clip(values)
        self.global_medians = {}
        for column in FEATURE_COLUMNS:
            non_missing = clipped[column].dropna()
            if non_missing.empty:
                raise ValueError(
                    f"Cannot fit imputation state: {column} has no finite training median."
                )
            median = non_missing.median()
            self.global_medians[column] = float(median)

        self.realm_medians = {}
        realm_key_values = np.asarray(realm_keys)
        for realm_key in dict.fromkeys(realm_keys):
            realm_rows = clipped.iloc[np.flatnonzero(realm_key_values == realm_key)]
            medians: dict[str, float] = {}
            for column in FEATURE_COLUMNS:
                non_missing = realm_rows[column].dropna()
                if not non_missing.empty:
                    medians[column] = float(non_missing.median())
            self.realm_medians[realm_key] = medians

        self._is_fitted = True
        return self

    def _clip(self, values: pd.DataFrame) -> pd.DataFrame:
        clipped = values.copy()
        for column, bounds in self.clip_bounds.items():
            clipped[column] = clipped[column].clip(
                lower=bounds["lower"], upper=bounds["upper"]
            )
        return clipped

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise ValueError("ModelPreprocessor must be fitted before transform.")
        values = self._clip(_numeric_features(frame))
        realm_keys = _realm_keys(values)

        for row_position, realm_key in enumerate(realm_keys):
            realm_medians = self.realm_medians.get(realm_key, {})
            for column in FEATURE_COLUMNS:
                if pd.isna(values.iat[row_position, values.columns.get_loc(column)]):
                    fill_value = realm_medians.get(
                        column, self.global_medians[column]
                    )
                    values.iat[row_position, values.columns.get_loc(column)] = (
                        fill_value
                    )

        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError("Preprocessing produced non-finite feature values.")
        return values.loc[:, FEATURE_COLUMNS]

    def to_state(self) -> dict[str, Any]:
        if not self._is_fitted:
            raise ValueError("Cannot serialize an unfitted ModelPreprocessor.")
        return {
            "schema_version": PREPROCESSING_SCHEMA_VERSION,
            "feature_columns": list(FEATURE_COLUMNS),
            "clip_quantiles": {
                "lower": self.lower_quantile,
                "upper": self.upper_quantile,
            },
            "non_clippable_columns": list(NON_CLIPPABLE_COLUMNS),
            "clip_bounds": {
                column: dict(bounds) for column, bounds in self.clip_bounds.items()
            },
            "imputation": {
                "strategy": IMPUTATION_STRATEGY,
                "realm_indicator_columns": list(REALM_INDICATOR_COLUMNS),
                "baseline_realm_key": BASELINE_REALM_KEY,
                "baseline_realm_name": BASELINE_REALM_NAME,
                "global_medians": dict(self.global_medians),
                "realm_medians": {
                    realm: dict(medians)
                    for realm, medians in self.realm_medians.items()
                },
            },
            "fit_scope": ARTIFACT_PREPROCESSING_FIT_SCOPE,
            "evaluation": {
                "preprocessing_fit_scope": EVALUATION_PREPROCESSING_FIT_SCOPE
            },
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> ModelPreprocessor:
        if not isinstance(state, Mapping):
            raise ValueError("Invalid preprocessing_state: expected an object.")
        schema_version = state.get("schema_version")
        if (
            type(schema_version) is not int
            or schema_version != PREPROCESSING_SCHEMA_VERSION
        ):
            raise ValueError(
                "Invalid preprocessing_state: unsupported or missing schema_version."
            )
        if state.get("feature_columns") != list(FEATURE_COLUMNS):
            raise ValueError(
                "Invalid preprocessing_state: feature_columns do not match the model schema."
            )
        if state.get("non_clippable_columns") != list(NON_CLIPPABLE_COLUMNS):
            raise ValueError(
                "Invalid preprocessing_state: non_clippable_columns do not match the model schema."
            )
        if state.get("fit_scope") != ARTIFACT_PREPROCESSING_FIT_SCOPE:
            raise ValueError(
                "Invalid preprocessing_state: fit_scope must be full_training_set."
            )
        evaluation = state.get("evaluation")
        if (
            not isinstance(evaluation, Mapping)
            or evaluation.get("preprocessing_fit_scope")
            != EVALUATION_PREPROCESSING_FIT_SCOPE
        ):
            raise ValueError(
                "Invalid preprocessing_state: evaluation preprocessing fit scope is missing or invalid."
            )

        quantiles = state.get("clip_quantiles")
        if not isinstance(quantiles, Mapping):
            raise ValueError(
                "Invalid preprocessing_state: clip_quantiles must be an object."
            )
        try:
            lower_quantile = cls._finite_number(
                quantiles["lower"], "clip_quantiles.lower"
            )
            upper_quantile = cls._finite_number(
                quantiles["upper"], "clip_quantiles.upper"
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid preprocessing_state: clip_quantiles are incomplete or invalid."
            ) from exc
        if (
            not np.isfinite([lower_quantile, upper_quantile]).all()
            or not 0 <= lower_quantile < upper_quantile <= 1
        ):
            raise ValueError(
                "Invalid preprocessing_state: clip_quantiles are outside the valid range."
            )

        clip_bounds = state.get("clip_bounds")
        expected_bounds = {
            column
            for column in FEATURE_COLUMNS
            if column not in set(NON_CLIPPABLE_COLUMNS)
        }
        if not isinstance(clip_bounds, Mapping) or set(clip_bounds) != expected_bounds:
            raise ValueError(
                "Invalid preprocessing_state: clip_bounds do not match the model schema."
            )
        validated_bounds: dict[str, dict[str, float]] = {}
        for column, bounds in clip_bounds.items():
            if not isinstance(bounds, Mapping):
                raise ValueError(
                    f"Invalid preprocessing_state: clip_bounds for {column} must be an object."
                )
            try:
                lower = cls._finite_number(
                    bounds["lower"], f"clip_bounds.{column}.lower"
                )
                upper = cls._finite_number(
                    bounds["upper"], f"clip_bounds.{column}.upper"
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid preprocessing_state: clip_bounds for {column} are incomplete."
                ) from exc
            if not np.isfinite([lower, upper]).all() or lower > upper:
                raise ValueError(
                    f"Invalid preprocessing_state: clip_bounds for {column} are invalid."
                )
            validated_bounds[column] = {"lower": lower, "upper": upper}

        imputation = state.get("imputation")
        if not isinstance(imputation, Mapping):
            raise ValueError("Invalid preprocessing_state: imputation must be an object.")
        if imputation.get("strategy") != IMPUTATION_STRATEGY:
            raise ValueError(
                "Invalid preprocessing_state: unsupported imputation strategy."
            )
        if imputation.get("realm_indicator_columns") != list(
            REALM_INDICATOR_COLUMNS
        ):
            raise ValueError(
                "Invalid preprocessing_state: Realm indicators do not match the model schema."
            )
        if (
            imputation.get("baseline_realm_key") != BASELINE_REALM_KEY
            or imputation.get("baseline_realm_name") != BASELINE_REALM_NAME
        ):
            raise ValueError(
                "Invalid preprocessing_state: baseline Realm does not match the model schema."
            )

        global_medians = cls._validated_medians(
            imputation.get("global_medians"),
            FEATURE_COLUMNS,
            "global_medians",
            exact=True,
        )
        realm_medians = imputation.get("realm_medians")
        if (
            not isinstance(realm_medians, Mapping)
            or BASELINE_REALM_KEY not in realm_medians
        ):
            raise ValueError(
                "Invalid preprocessing_state: realm_medians must include the baseline Realm."
            )
        validated_realm_medians = {}
        for realm_key, medians in realm_medians.items():
            if not isinstance(realm_key, str) or not isinstance(medians, Mapping):
                raise ValueError(
                    "Invalid preprocessing_state: Realm median entries must be objects."
                )
            validated_realm_medians[realm_key] = cls._validated_medians(
                medians,
                FEATURE_COLUMNS,
                f"realm_medians[{realm_key}]",
                exact=False,
            )

        preprocessor = cls(lower_quantile, upper_quantile)
        preprocessor.clip_bounds = validated_bounds
        preprocessor.global_medians = global_medians
        preprocessor.realm_medians = validated_realm_medians
        preprocessor._is_fitted = True
        return preprocessor

    @staticmethod
    def _validated_medians(
        values: Any,
        expected_columns: tuple[str, ...],
        field_name: str,
        *,
        exact: bool,
    ) -> dict[str, float]:
        if not isinstance(values, Mapping):
            raise ValueError(
                f"Invalid preprocessing_state: {field_name} must be an object."
            )
        keys = set(values)
        expected = set(expected_columns)
        if (exact and keys != expected) or (not exact and not keys.issubset(expected)):
            raise ValueError(
                f"Invalid preprocessing_state: {field_name} columns do not match the model schema."
            )
        validated: dict[str, float] = {}
        for column, value in values.items():
            if not isinstance(column, str):
                raise ValueError(
                    f"Invalid preprocessing_state: {field_name} keys must be strings."
                )
            try:
                median = ModelPreprocessor._finite_number(
                    value, f"{field_name}.{column}"
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid preprocessing_state: {field_name}.{column} must be numeric."
                ) from exc
            validated[column] = median
        if exact and len(validated) != len(expected_columns):
            raise ValueError(
                f"Invalid preprocessing_state: {field_name} is incomplete."
            )
        return validated

    @staticmethod
    def _finite_number(value: Any, field_name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ValueError(f"{field_name} must be numeric.")
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"{field_name} must be finite.")
        return number
