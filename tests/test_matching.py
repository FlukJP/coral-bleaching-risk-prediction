import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training.matching import match_auxiliary_features


ROOT = Path(__file__).resolve().parents[1]


def _observation(latitude=0.0, longitude=0.0, date="2020-01-01"):
    return pd.DataFrame(
        {
            "Latitude_Degrees": [latitude],
            "Longitude_Degrees": [longitude],
            "Date": [date],
        }
    )


def _source(latitude=0.0, longitude=0.0, date="2020-01-01", value=1.0):
    return pd.DataFrame(
        {
            "Latitude": [latitude],
            "Longitude": [longitude],
            "Date": [date],
            "SST (\N{DEGREE SIGN}C)": [28.0],
            "Marine Heatwave": [value],
            "pH Level": [8.1],
        }
    )


def _match(observations, source):
    heatwave, ph, count = match_auxiliary_features(observations, source)
    return heatwave, ph, count


@pytest.mark.parametrize(
    ("days_away", "expected_count"),
    [(30, 1), (31, 0)],
)
def test_date_window_boundary(days_away, expected_count):
    observations = _observation()
    source_date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=days_away)
    source = _source(date=source_date)

    _, _, count = _match(observations, source)

    assert count == expected_count


@pytest.mark.parametrize(
    ("distance_km", "expected_count"),
    [(499, 1), (502, 0)],
)
def test_distance_window_boundary(distance_km, expected_count):
    latitude = distance_km / 6371.0 * 180.0 / np.pi
    observations = _observation()
    source = _source(latitude=latitude)

    _, _, count = _match(observations, source)

    assert count == expected_count


def test_equal_distance_candidates_keep_balltree_first_candidate():
    observations = _observation()
    source = pd.concat(
        [
            _source(longitude=-1.0, value=0.0),
            _source(longitude=1.0, value=1.0),
        ],
        ignore_index=True,
    )

    heatwave, ph, count = _match(observations, source)

    assert count == 1
    assert heatwave[0] == 0.0
    assert ph[0] == pytest.approx(8.1)


def test_nearest_date_eligible_candidate_is_selected():
    observations = _observation()
    source = pd.concat(
        [
            _source(latitude=0.01, date="2020-02-01", value=0.0),
            _source(latitude=0.02, date="2020-01-02", value=1.0),
            _source(latitude=0.03, date="2020-01-01", value=0.0),
        ],
        ignore_index=True,
    )

    heatwave, _, count = _match(observations, source)

    assert count == 1
    assert heatwave[0] == 1.0


@pytest.mark.parametrize(
    "observations",
    [
        _observation(date="not-a-date"),
        _observation(latitude=np.nan),
        _observation(latitude=np.inf),
        _observation(longitude=np.nan),
        _observation(longitude=-np.inf),
    ],
)
def test_invalid_observation_date_or_coordinates_are_unmatched(observations):
    heatwave, ph, count = _match(observations, _source())

    assert count == 0
    assert np.isnan(heatwave[0])
    assert np.isnan(ph[0])


@pytest.mark.parametrize(
    "source",
    [
        _source(latitude=np.nan),
        _source(longitude=np.nan),
        _source(date="not-a-date"),
    ],
)
def test_invalid_source_date_or_coordinates_are_unmatched(source):
    heatwave, ph, count = _match(_observation(), source)

    assert count == 0
    assert np.isnan(heatwave[0])
    assert np.isnan(ph[0])


def test_no_eligible_pair_leaves_auxiliary_values_missing():
    heatwave, ph, count = _match(_observation(), _source(latitude=10.0))

    assert count == 0
    assert np.isnan(heatwave[0])
    assert np.isnan(ph[0])


def _load_raw_matching_inputs():
    observations = pd.read_csv(
        ROOT / "data" / "global_bleaching_environmental.csv",
        low_memory=False,
    ).replace(["nd", "ND", "na", "NA", "None", "none", ""], np.nan)
    source = pd.read_csv(
        ROOT / "data" / "realistic_ocean_climate_dataset.csv",
        low_memory=False,
    ).replace(["nd", "ND", "na", "NA", "None", "none", ""], np.nan)

    for column in ("Latitude_Degrees", "Longitude_Degrees", "Percent_Bleaching"):
        observations[column] = pd.to_numeric(observations[column], errors="coerce")
    observations["Date"] = pd.to_datetime(observations["Date"], errors="coerce")
    observations = observations.dropna(subset=["Percent_Bleaching"]).reset_index(
        drop=True
    )

    source["Date"] = pd.to_datetime(source["Date"], errors="coerce")
    source["Marine Heatwave"] = source["Marine Heatwave"].astype(str).str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0}
    )
    return observations, source


def test_real_data_matches_pre_move_reference():
    reference = json.loads(
        (ROOT / "tests" / "fixtures" / "matching_reference.json").read_text(
            encoding="utf-8"
        )
    )
    for filename, expected_hash in reference["source_sha256"].items():
        source_hash = hashlib.sha256(
            (ROOT / "data" / filename).read_bytes()
        ).hexdigest()
        assert source_hash == expected_hash

    observations, source = _load_raw_matching_inputs()
    heatwave, ph, match_count = match_auxiliary_features(observations, source)

    output = pd.DataFrame(
        {
            "row_index": range(len(observations)),
            "Aux_Marine_Heatwave": heatwave,
            "Aux_pH_Level": ph,
        }
    )
    output_hash = hashlib.sha256(
        output.to_csv(
            index=False,
            lineterminator="\n",
            float_format="%.17g",
        ).encode("utf-8")
    ).hexdigest()

    assert len(observations) == reference["target_valid_rows"]
    assert match_count == reference["matched_rows"]
    assert output_hash == reference["output_sha256"]
