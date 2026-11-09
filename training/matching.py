import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


def match_auxiliary_features(
    observations: pd.DataFrame,
    source: pd.DataFrame,
    max_date_difference_days: int = 30,
    max_distance_km: int = 500,
) -> tuple[np.ndarray, np.ndarray, int]:
    source_clean = source.dropna(
        subset=["Latitude", "Longitude", "SST (\N{DEGREE SIGN}C)"]
    ).copy()
    source_clean["Date"] = pd.to_datetime(source_clean["Date"], errors="coerce")
    source_clean = source_clean.dropna(subset=["Date"])

    observation_dates = pd.to_datetime(
        observations["Date"], errors="coerce"
    ).to_numpy(dtype="datetime64[ns]")
    observation_coordinates = observations[
        ["Latitude_Degrees", "Longitude_Degrees"]
    ].to_numpy(dtype=float)
    valid_observations = (
        ~pd.isna(observation_dates)
        & np.isfinite(observation_coordinates).all(axis=1)
    )
    valid_row_indices = np.flatnonzero(valid_observations)

    auxiliary_heatwave = np.full(len(observations), np.nan)
    auxiliary_ph = np.full(len(observations), np.nan)
    match_count = 0

    if not source_clean.empty and valid_row_indices.size:
        source_coordinates = np.radians(
            source_clean[["Latitude", "Longitude"]].to_numpy(dtype=float)
        )
        tree = BallTree(source_coordinates, metric="haversine")
        candidate_indices, _ = tree.query_radius(
            np.radians(observation_coordinates[valid_row_indices]),
            r=max_distance_km / 6371.0,
            return_distance=True,
            sort_results=True,
        )
        source_dates = source_clean["Date"].to_numpy(dtype="datetime64[ns]")
        source_heatwave = source_clean["Marine Heatwave"].to_numpy()
        source_ph = source_clean["pH Level"].to_numpy()

        for row_index, candidates in zip(valid_row_indices, candidate_indices):
            if not len(candidates):
                continue

            date_differences = np.abs(
                (source_dates[candidates] - observation_dates[row_index])
                / np.timedelta64(1, "D")
            )
            eligible_positions = np.flatnonzero(
                date_differences <= max_date_difference_days
            )
            if not eligible_positions.size:
                continue

            source_index = candidates[eligible_positions[0]]
            auxiliary_heatwave[row_index] = source_heatwave[source_index]
            auxiliary_ph[row_index] = source_ph[source_index]
            match_count += 1

    return auxiliary_heatwave, auxiliary_ph, match_count
