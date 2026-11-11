from pathlib import Path

import pandas as pd
from pandas.api.types import is_bool_dtype, is_float_dtype, is_integer_dtype

from model_preprocessing import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA,
    MODEL_READY_COLUMNS,
    REQUIRED_NON_NULL_COLUMNS,
    TARGET_NAME,
)


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SCHEMA = {
    **FEATURE_SCHEMA,
    TARGET_NAME: "float",
}
EXPECTED_SCHEMA = {
    column: EXPECTED_SCHEMA[column] for column in MODEL_READY_COLUMNS
}
ALLOWED_MISSING_COLUMNS = set(FEATURE_COLUMNS) - set(REQUIRED_NON_NULL_COLUMNS)


def test_model_ready_schema_and_non_null_columns():
    data = pd.read_csv(ROOT / "data" / "bleaching_model_ready.csv")

    assert list(data.columns) == list(EXPECTED_SCHEMA)
    assert len(FEATURE_COLUMNS) == 61
    assert data[list(REQUIRED_NON_NULL_COLUMNS)].notna().all().all()
    assert set(data.columns[data.isna().any()]).issubset(ALLOWED_MISSING_COLUMNS)
    for column, expected_type in EXPECTED_SCHEMA.items():
        dtype = data[column].dtype
        if expected_type == "float":
            assert is_float_dtype(dtype), column
        elif expected_type == "integer":
            assert is_integer_dtype(dtype), column
        else:
            assert is_bool_dtype(dtype), column
