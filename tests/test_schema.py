from pathlib import Path

import pandas as pd
from pandas.api.types import is_bool_dtype, is_float_dtype, is_integer_dtype


ROOT = Path(__file__).resolve().parents[1]

# Keep the model-ready schema contract in this single mapping for easy updates.
EXPECTED_SCHEMA = {
    "Latitude_Degrees": "float",
    "Longitude_Degrees": "float",
    "Distance_to_Shore": "float",
    "Turbidity": "float",
    "Cyclone_Frequency": "float",
    "Date_Day": "integer",
    "Date_Month": "integer",
    "Date_Year": "integer",
    "Depth_m": "float",
    "Percent_Bleaching": "float",
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


def test_model_ready_schema_and_non_null_columns():
    data = pd.read_csv(ROOT / "data" / "bleaching_model_ready.csv")

    assert list(data.columns) == list(EXPECTED_SCHEMA)
    assert data.notna().all().all()
    for column, expected_type in EXPECTED_SCHEMA.items():
        dtype = data[column].dtype
        if expected_type == "float":
            assert is_float_dtype(dtype), column
        elif expected_type == "integer":
            assert is_integer_dtype(dtype), column
        else:
            assert is_bool_dtype(dtype), column
