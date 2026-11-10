import hashlib
import json
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path

import pytest

from model_metadata import (
    FRESH,
    MISSING,
    MISMATCH,
    STALE,
    UNVERIFIABLE,
    SCHEMA_VERSION,
    STALE_AFTER_DAYS,
    get_dataset_status,
    get_model_data_status,
    write_dataset_metadata,
    write_model_metadata,
)


def _write_metadata_files(tmp_path, model_csv_hash=None, model_metadata=True):
    csv_path = tmp_path / "bleaching_model_ready.csv"
    csv_path.write_bytes(b"model-ready")
    csv_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    dataset_metadata_path = tmp_path / "metadata.json"
    dataset_metadata_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "csv_file": csv_path.name,
                "csv_sha256": csv_hash,
                "dependency_versions": {"pandas": "2.3.3"},
                "coverage": {
                    "rows_total": 3,
                    "matched_rows": 2,
                    "match_denominator": 3,
                    "matched_share": 2 / 3,
                    "imputed_share_by_feature": {"Aux_pH_Level": 1 / 3},
                },
            }
        ),
        encoding="utf-8",
    )
    model_directory = tmp_path / "stacking_ensemble"
    model_directory.mkdir()
    model_metadata_path = model_directory / "metadata.json"
    if model_metadata:
        model_metadata_path.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "training_csv_hash": model_csv_hash or csv_hash,
                    "dependency_versions": {"scikit-learn": "1.8.0"},
                    "evaluation": {
                        "method": "GroupKFold",
                        "n_folds": 4,
                        "n_validation_rows": 3,
                        "metrics": {"severity": {"RMSE_mean": 12.3}},
                        "evaluated_at": datetime.now(timezone.utc).isoformat(),
                    },
                    "coverage": {
                        "rows_total": 3,
                        "matched_rows": 2,
                        "match_denominator": 3,
                        "matched_share": 2 / 3,
                        "imputed_share_by_feature": {"Aux_pH_Level": 1 / 3},
                    },
                }
            ),
            encoding="utf-8",
        )
    return csv_path, dataset_metadata_path, model_metadata_path


def test_fresh_dataset_and_model_match(tmp_path):
    csv_path, dataset_meta, model_meta = _write_metadata_files(tmp_path)

    result = get_model_data_status(
        csv_path, dataset_meta, model_meta, datetime.now(timezone.utc)
    )

    assert result.status == FRESH
    assert result.age_days == 0
    assert result.model_metadata["evaluation"]["method"] == "GroupKFold"


def test_stale_status_reports_file_age(tmp_path):
    csv_path, dataset_meta, model_meta = _write_metadata_files(tmp_path)
    old_mtime = datetime.now(timezone.utc) - timedelta(days=STALE_AFTER_DAYS + 1)
    os.utime(csv_path, (old_mtime.timestamp(), old_mtime.timestamp()))

    result = get_model_data_status(
        csv_path, dataset_meta, model_meta, datetime.now(timezone.utc)
    )

    assert result.status == STALE
    assert f"{STALE_AFTER_DAYS + 1} วัน" in result.message
    assert "ข้อมูลสิ่งแวดล้อม" not in result.message


def test_file_age_at_threshold_is_not_stale(tmp_path):
    csv_path, dataset_meta, model_meta = _write_metadata_files(tmp_path)
    mtime = datetime.now(timezone.utc) - timedelta(days=STALE_AFTER_DAYS)
    os.utime(csv_path, (mtime.timestamp(), mtime.timestamp()))

    result = get_model_data_status(
        csv_path, dataset_meta, model_meta, datetime.now(timezone.utc)
    )

    assert result.status == FRESH
    assert result.age_days == STALE_AFTER_DAYS


def test_model_csv_hash_mismatch_is_per_model(tmp_path):
    csv_path, dataset_meta, model_meta = _write_metadata_files(
        tmp_path, model_csv_hash="0" * 64
    )

    result = get_model_data_status(
        csv_path, dataset_meta, model_meta, datetime.now(timezone.utc)
    )

    assert result.status == MISMATCH
    assert "training_csv_hash" in result.message


def test_model_statuses_are_checked_independently(tmp_path):
    csv_path, dataset_meta, stacking_metadata = _write_metadata_files(tmp_path)
    mlp_directory = tmp_path / "custom_residual_mlp"
    mlp_directory.mkdir()
    mlp_metadata = mlp_directory / "metadata.json"
    mlp_payload = json.loads(stacking_metadata.read_text(encoding="utf-8"))
    mlp_payload["training_csv_hash"] = "0" * 64
    mlp_metadata.write_text(json.dumps(mlp_payload), encoding="utf-8")

    stacking_status = get_model_data_status(
        csv_path, dataset_meta, stacking_metadata, datetime.now(timezone.utc)
    )
    mlp_status = get_model_data_status(
        csv_path, dataset_meta, mlp_metadata, datetime.now(timezone.utc)
    )

    assert stacking_status.status == FRESH
    assert mlp_status.status == MISMATCH


def test_current_hash_is_checked_even_if_data_metadata_hash_is_stale(tmp_path):
    csv_path, dataset_meta, model_meta = _write_metadata_files(tmp_path)
    model_data = json.loads(model_meta.read_text(encoding="utf-8"))
    model_data["training_csv_hash"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    model_meta.write_text(json.dumps(model_data), encoding="utf-8")
    data = json.loads(dataset_meta.read_text(encoding="utf-8"))
    data["csv_sha256"] = "0" * 64
    dataset_meta.write_text(json.dumps(data), encoding="utf-8")

    result = get_model_data_status(
        csv_path, dataset_meta, model_meta, datetime.now(timezone.utc)
    )

    assert result.status == MISMATCH
    assert result.model_metadata["training_csv_hash"] == result.csv_sha256


def test_missing_csv_has_missing_status(tmp_path):
    csv_path = tmp_path / "missing.csv"

    result = get_dataset_status(csv_path, tmp_path / "metadata.json")

    assert result.status == MISSING


def test_missing_metadata_is_unverifiable_not_fresh(tmp_path):
    csv_path, dataset_meta, model_meta = _write_metadata_files(
        tmp_path, model_metadata=False
    )

    result = get_model_data_status(
        csv_path, dataset_meta, model_meta, datetime.now(timezone.utc)
    )

    assert result.status == UNVERIFIABLE


def test_invalid_or_incomplete_json_metadata_is_unverifiable(tmp_path):
    csv_path, dataset_meta, model_meta = _write_metadata_files(tmp_path)
    model_meta.write_text("{broken", encoding="utf-8")

    result = get_model_data_status(
        csv_path, dataset_meta, model_meta, datetime.now(timezone.utc)
    )

    assert result.status == UNVERIFIABLE

    model_meta.write_text(
        json.dumps({"schema_version": SCHEMA_VERSION}), encoding="utf-8"
    )
    result = get_model_data_status(
        csv_path, dataset_meta, model_meta, datetime.now(timezone.utc)
    )
    assert result.status == UNVERIFIABLE


def test_metadata_writers_preserve_schema_and_link_csv_hash(tmp_path):
    csv_path = tmp_path / "bleaching_model_ready.csv"
    csv_path.write_bytes(b"training-data")
    source_path = tmp_path / "source.csv"
    source_path.write_bytes(b"source")
    data_metadata_path = write_dataset_metadata(
        csv_path,
        [source_path],
        row_count=2,
        matched_rows=1,
        imputed_values={"Aux_pH_Level": [False, True]},
    )
    model_directory = tmp_path / "model"
    evaluation = {
        "method": "GroupKFold",
        "n_folds": 2,
        "n_validation_rows": 2,
        "metrics": {"severity": {"RMSE": 1.25}},
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    model_metadata_path = write_model_metadata(csv_path, model_directory, evaluation)

    data_metadata = json.loads(data_metadata_path.read_text(encoding="utf-8"))
    model_metadata = json.loads(model_metadata_path.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert data_metadata["schema_version"] == SCHEMA_VERSION
    assert data_metadata["csv_sha256"] == expected_hash
    assert data_metadata["coverage"]["imputed_share_by_feature"]["Aux_pH_Level"] == 0.5
    assert model_metadata["training_csv_hash"] == expected_hash
    assert model_metadata["evaluation"] == evaluation


def test_model_metadata_writer_rejects_stale_dataset_hash(tmp_path):
    csv_path, dataset_meta, _ = _write_metadata_files(tmp_path)
    csv_path.write_bytes(b"changed")
    evaluation = {
        "method": "GroupKFold",
        "n_folds": 2,
        "n_validation_rows": 2,
        "metrics": {"RMSE": 1.0},
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    with pytest.raises(ValueError, match="hash does not match"):
        write_model_metadata(csv_path, tmp_path / "model", evaluation)


def test_dashboard_does_not_claim_unverified_accuracy_or_production_status():
    dashboard = (
        Path(__file__).resolve().parents[1] / "streamlit_app.py"
    ).read_text(encoding="utf-8").lower()

    assert "high-accuracy" not in dashboard
    assert "production model" not in dashboard
    assert "validation rows across folds" in dashboard
    assert "auxiliary coverage" in dashboard
