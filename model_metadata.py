from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STALE_AFTER_DAYS = 30
SCHEMA_VERSION = 1
DEPENDENCY_NAMES = (
    "ipython",
    "ipykernel",
    "joblib",
    "lightgbm",
    "numpy",
    "optuna",
    "pandas",
    "scikit-learn",
    "streamlit",
    "torch",
    "xgboost",
)

MISSING = "Missing"
STALE = "Stale"
MISMATCH = "Mismatch"
FRESH = "Fresh"
UNVERIFIABLE = "ตรวจสอบไม่ได้"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class DataStatus:
    status: str
    message: str
    age_days: int | None = None
    metadata: dict[str, Any] | None = None
    model_metadata: dict[str, Any] | None = None
    csv_sha256: str | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_versions() -> dict[str, str]:
    versions = {}
    for package in DEPENDENCY_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Required package {package!r} is not installed; cannot record metadata."
            ) from exc
    return versions


def _json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if hasattr(value, "item"):
        return _json_compatible(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(
            _json_compatible(payload),
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        file.write("\n")


def write_dataset_metadata(
    csv_path: Path,
    source_paths: list[Path],
    row_count: int,
    matched_rows: int,
    imputed_values: dict[str, Any],
) -> Path:
    denominator = row_count
    imputed_shares = {
        feature: float(sum(bool(value) for value in values) / denominator)
        if denominator
        else 0.0
        for feature, values in imputed_values.items()
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "csv_file": csv_path.name,
        "csv_sha256": sha256_file(csv_path),
        "source_csv_hashes": {
            path.name: sha256_file(path) for path in source_paths
        },
        "dependency_versions": dependency_versions(),
        "coverage": {
            "rows_total": row_count,
            "matched_rows": matched_rows,
            "match_denominator": denominator,
            "matched_share": float(matched_rows / denominator) if denominator else 0.0,
            "imputed_share_by_feature": imputed_shares,
        },
    }
    metadata_path = csv_path.with_name("metadata.json")
    _write_json(metadata_path, metadata)
    return metadata_path


def write_model_metadata(
    csv_path: Path,
    model_directory: Path,
    evaluation: dict[str, Any],
) -> Path:
    dataset_metadata_path = csv_path.with_name("metadata.json")
    dataset_metadata = _read_json(dataset_metadata_path)
    if not _valid_dataset_metadata(dataset_metadata):
        raise ValueError(
            "Dataset metadata is incomplete or unsupported; "
            "cannot write model metadata."
        )
    current_hash = sha256_file(csv_path)
    if dataset_metadata.get("csv_sha256") != current_hash:
        raise ValueError(
            "Dataset metadata hash does not match the training CSV; "
            "cannot write model metadata."
        )
    _validate_evaluation(evaluation)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_csv_hash": current_hash,
        "dependency_versions": dependency_versions(),
        "evaluation": evaluation,
        "coverage": dataset_metadata["coverage"],
    }
    metadata_path = model_directory / "metadata.json"
    _write_json(metadata_path, metadata)
    return metadata_path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError("Metadata must contain a JSON object.")
    return value


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _valid_coverage(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    rows_total = value.get("rows_total")
    matched_rows = value.get("matched_rows")
    denominator = value.get("match_denominator")
    matched_share = value.get("matched_share")
    imputed_shares = value.get("imputed_share_by_feature")
    return (
        isinstance(rows_total, int)
        and rows_total >= 0
        and isinstance(matched_rows, int)
        and isinstance(denominator, int)
        and denominator >= 0
        and 0 <= matched_rows <= denominator
        and isinstance(matched_share, (int, float))
        and 0 <= matched_share <= 1
        and isinstance(imputed_shares, dict)
        and all(
            isinstance(share, (int, float)) and 0 <= share <= 1
            for share in imputed_shares.values()
        )
    )


def _valid_dataset_metadata(value: dict[str, Any]) -> bool:
    return (
        value.get("schema_version") == SCHEMA_VERSION
        and _parse_timestamp(value.get("created_at")) is not None
        and _is_sha256(value.get("csv_sha256"))
        and isinstance(value.get("dependency_versions"), dict)
        and _valid_coverage(value.get("coverage"))
    )


def _valid_evaluation(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        isinstance(value.get("method"), str)
        and bool(value["method"])
        and isinstance(value.get("n_folds"), int)
        and value["n_folds"] > 0
        and isinstance(value.get("n_validation_rows"), int)
        and value["n_validation_rows"] > 0
        and isinstance(value.get("metrics"), dict)
        and bool(value["metrics"])
        and _parse_timestamp(value.get("evaluated_at")) is not None
    )


def _validate_evaluation(value: Any) -> None:
    if not _valid_evaluation(value):
        raise ValueError(
            "Evaluation metadata must include method, n_folds, "
            "n_validation_rows, metrics, and evaluated_at."
        )


def _valid_model_metadata(value: dict[str, Any]) -> bool:
    return (
        value.get("schema_version") == SCHEMA_VERSION
        and _parse_timestamp(value.get("created_at")) is not None
        and _is_sha256(value.get("training_csv_hash"))
        and isinstance(value.get("dependency_versions"), dict)
        and _valid_coverage(value.get("coverage"))
        and _valid_evaluation(value.get("evaluation"))
    )


def get_dataset_status(
    csv_path: Path,
    dataset_metadata_path: Path,
    now: datetime | None = None,
) -> DataStatus:
    if not csv_path.is_file():
        return DataStatus(MISSING, "ไฟล์ model-ready CSV ไม่พบ")
    if not dataset_metadata_path.is_file():
        return DataStatus(UNVERIFIABLE, "ไม่พบ metadata.json ของข้อมูล")
    try:
        metadata = _read_json(dataset_metadata_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return DataStatus(UNVERIFIABLE, "metadata.json ของข้อมูลอ่านไม่ได้")
    if not _valid_dataset_metadata(metadata):
        return DataStatus(UNVERIFIABLE, "metadata.json ของข้อมูลไม่ครบหรือไม่รองรับ")
    try:
        current_hash = sha256_file(csv_path)
        modified_at = datetime.fromtimestamp(csv_path.stat().st_mtime, timezone.utc)
    except OSError:
        return DataStatus(UNVERIFIABLE, "ตรวจไฟล์ model-ready CSV ไม่ได้")
    if current_hash != metadata["csv_sha256"]:
        return DataStatus(
            MISMATCH,
            "hash ของ CSV ไม่ตรงกับ metadata ของข้อมูล",
            metadata=metadata,
            csv_sha256=current_hash,
        )
    age = (now or datetime.now(timezone.utc)).astimezone(timezone.utc) - modified_at
    if age.total_seconds() < 0:
        return DataStatus(UNVERIFIABLE, "เวลาแก้ไขไฟล์ model-ready CSV ไม่ถูกต้อง", metadata=metadata)
    age_days = age.days
    if age_days > STALE_AFTER_DAYS:
        return DataStatus(
            STALE,
            f"ไฟล์ model-ready CSV มีอายุ {age_days} วัน (เกณฑ์ {STALE_AFTER_DAYS} วัน)",
            age_days,
            metadata,
            csv_sha256=current_hash,
        )
    return DataStatus(
        FRESH,
        f"ไฟล์ model-ready CSV มีอายุ {age_days} วัน",
        age_days,
        metadata,
        csv_sha256=current_hash,
    )


def get_model_data_status(
    csv_path: Path,
    dataset_metadata_path: Path,
    model_metadata_path: Path,
    now: datetime | None = None,
) -> DataStatus:
    dataset_status = get_dataset_status(csv_path, dataset_metadata_path, now)
    if dataset_status.status == MISSING:
        return dataset_status
    if not model_metadata_path.is_file():
        if not dataset_metadata_path.is_file():
            message = (
                "ไม่พบ metadata.json ของข้อมูลและโมเดล "
                f"({model_metadata_path.parent.name})"
            )
        else:
            message = (
                f"ไม่พบ metadata.json ของโมเดล ({model_metadata_path.parent.name})"
            )
        return DataStatus(
            UNVERIFIABLE,
            message,
            dataset_status.age_days,
            dataset_status.metadata,
            csv_sha256=dataset_status.csv_sha256,
        )
    if dataset_status.status == UNVERIFIABLE:
        return dataset_status
    try:
        model_metadata = _read_json(model_metadata_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return DataStatus(
            UNVERIFIABLE,
            f"metadata.json ของโมเดล ({model_metadata_path.parent.name}) อ่านไม่ได้",
            dataset_status.age_days,
            dataset_status.metadata,
        )
    if not _valid_model_metadata(model_metadata):
        return DataStatus(
            UNVERIFIABLE,
            f"metadata.json ของโมเดล ({model_metadata_path.parent.name}) ไม่ครบหรือไม่รองรับ",
            dataset_status.age_days,
            dataset_status.metadata,
        )
    if model_metadata["training_csv_hash"] != dataset_status.csv_sha256:
        return DataStatus(
            MISMATCH,
            "hash ของ CSV ปัจจุบันไม่ตรงกับ training_csv_hash ของโมเดล",
            dataset_status.age_days,
            dataset_status.metadata,
            model_metadata,
            dataset_status.csv_sha256,
        )
    if dataset_status.status == MISMATCH:
        return DataStatus(
            MISMATCH,
            dataset_status.message,
            dataset_status.age_days,
            dataset_status.metadata,
            model_metadata,
            dataset_status.csv_sha256,
        )
    if dataset_status.status == STALE:
        return DataStatus(
            STALE,
            dataset_status.message,
            dataset_status.age_days,
            dataset_status.metadata,
            model_metadata,
            dataset_status.csv_sha256,
        )
    return DataStatus(
        FRESH,
        dataset_status.message,
        dataset_status.age_days,
        dataset_status.metadata,
        model_metadata,
        dataset_status.csv_sha256,
    )
