from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "bleaching_model_ready.csv"
STACKING_ARTIFACT_PATH = ROOT_DIR / "model" / "stacking_ensemble" / "Stacking_Ensemble_model.joblib"
MLP_ARTIFACT_PATH = ROOT_DIR / "model" / "custom_residual_mlp" / "custom_residual_mlp_artifact.pt"


class TabularAttention(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.Tanh(),
            nn.Linear(in_features // 2, in_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_weights = self.attention(x)
        return x * attn_weights


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.ln1(self.fc1(x)))
        out = self.dropout(out)
        out = self.ln2(self.fc2(out))
        out += residual
        return F.relu(out)


class CoralResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_blocks: int = 3, dropout: float = 0.2):
        super().__init__()
        self.attention = TabularAttention(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_head(x)


@dataclass
class PreparedInput:
    original: pd.DataFrame
    stacking: pd.DataFrame
    mlp: pd.DataFrame
    fill_notes: list[str]


def load_reference_features() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if "Percent_Bleaching" in df.columns:
        df = df.drop(columns=["Percent_Bleaching"])
    return df


def load_stacking_artifact() -> dict:
    return joblib.load(STACKING_ARTIFACT_PATH)


def load_mlp_artifact() -> tuple[dict, CoralResidualMLP]:
    try:
        artifact = torch.load(MLP_ARTIFACT_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        artifact = torch.load(MLP_ARTIFACT_PATH, map_location="cpu")

    config = artifact["config"]
    model = CoralResidualMLP(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_blocks=config["num_blocks"],
        dropout=config["dropout"],
    )
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()
    return artifact, model


def _default_value(series: pd.Series):
    if _is_boolean_like(series):
        mode = series.mode(dropna=True)
        return bool(_coerce_bool(mode.iloc[0])) if not mode.empty else False
    if pd.api.types.is_numeric_dtype(series):
        return float(series.median())
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else ""


def _coerce_bool(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    truthy = {"1", "true", "t", "yes", "y"}
    falsy = {"0", "false", "f", "no", "n"}
    if normalized in truthy:
        return True
    if normalized in falsy:
        return False
    return np.nan


def _is_boolean_like(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    non_null = series.dropna()
    if non_null.empty:
        return False
    normalized = {str(value).strip().lower() for value in non_null.unique()}
    allowed = {"1", "0", "true", "false", "t", "f", "yes", "no", "y", "n"}
    return normalized.issubset(allowed)


def prepare_input_frame(input_df: pd.DataFrame, reference_df: pd.DataFrame, stacking_artifact: dict, mlp_artifact: dict) -> PreparedInput:
    if input_df.empty:
        raise ValueError("The input table is empty.")

    df = input_df.copy()
    original_features = stacking_artifact.get("original_feature_names", stacking_artifact["feature_names"])
    sanitized_features = stacking_artifact["feature_names"]
    mlp_features = mlp_artifact["config"]["feature_names"]

    canonical = pd.DataFrame(index=df.index)
    missing_columns: list[str] = []
    for original_name, sanitized_name in zip(original_features, sanitized_features):
        if original_name in df.columns:
            canonical[original_name] = df[original_name]
        elif sanitized_name in df.columns:
            canonical[original_name] = df[sanitized_name]
        else:
            missing_columns.append(original_name)

    if missing_columns:
        preview = ", ".join(missing_columns[:8])
        suffix = " ..." if len(missing_columns) > 8 else ""
        raise ValueError(
            "Missing required feature columns. "
            f"Expected the processed schema from bleaching_model_ready.csv. Missing: {preview}{suffix}"
        )

    fill_notes: list[str] = []
    for column in original_features:
        reference_col = reference_df[column]
        default_value = _default_value(reference_col)

        if _is_boolean_like(reference_col):
            canonical[column] = canonical[column].map(_coerce_bool)
            missing_mask = canonical[column].isna()
            if missing_mask.any():
                fill_notes.append(f"{column}: filled {int(missing_mask.sum())} invalid boolean value(s) with {default_value}")
                canonical.loc[missing_mask, column] = default_value
            canonical[column] = canonical[column].astype(int)
        elif pd.api.types.is_numeric_dtype(reference_col):
            canonical[column] = pd.to_numeric(canonical[column], errors="coerce")
            missing_mask = canonical[column].isna()
            if missing_mask.any():
                fill_notes.append(f"{column}: filled {int(missing_mask.sum())} missing numeric value(s) with {default_value:.4f}")
                canonical.loc[missing_mask, column] = default_value
            canonical[column] = canonical[column].astype(float)
        else:
            missing_mask = canonical[column].isna() | (canonical[column].astype(str).str.strip() == "")
            if missing_mask.any():
                fill_notes.append(f"{column}: filled {int(missing_mask.sum())} missing text value(s) with '{default_value}'")
                canonical.loc[missing_mask, column] = default_value

    stacking_df = canonical[original_features].rename(columns=dict(zip(original_features, sanitized_features)))
    mlp_df = canonical[mlp_features].copy()

    return PreparedInput(
        original=canonical[original_features].copy(),
        stacking=stacking_df,
        mlp=mlp_df,
        fill_notes=fill_notes,
    )


def predict_stacking_ensemble(stacking_artifact: dict, stacking_frame: pd.DataFrame) -> pd.DataFrame:
    stacking_model = stacking_artifact["stacking_model"]
    classifier = stacking_artifact["two_stage_classifier"]
    regressor = stacking_artifact["two_stage_regressor"]
    threshold = float(stacking_artifact["two_stage_threshold"])
    blend_alpha = float(stacking_artifact["blend_alpha"])

    stacking_pred = np.expm1(stacking_model.predict(stacking_frame))
    stacking_pred = np.clip(stacking_pred, 0, 100)

    risk_proba = classifier.predict_proba(stacking_frame)[:, 1]
    risk_flag = (risk_proba >= threshold).astype(int)

    two_stage_pred = np.zeros(len(stacking_frame), dtype=float)
    positive_mask = risk_flag == 1
    if positive_mask.any():
        severity_pred = np.expm1(regressor.predict(stacking_frame.loc[positive_mask]))
        two_stage_pred[positive_mask] = np.clip(severity_pred, 0, 100)

    hybrid_pred = blend_alpha * stacking_pred + (1.0 - blend_alpha) * two_stage_pred
    hybrid_pred = np.clip(hybrid_pred, 0, 100)

    return pd.DataFrame(
        {
            "stacking_prediction": stacking_pred,
            "bleaching_risk_probability": risk_proba,
            "bleaching_risk_flag": risk_flag,
            "two_stage_prediction": two_stage_pred,
            "hybrid_prediction": hybrid_pred,
        },
        index=stacking_frame.index,
    )


def predict_custom_residual_mlp(mlp_artifact: dict, mlp_model: CoralResidualMLP, mlp_frame: pd.DataFrame) -> pd.Series:
    scaler = mlp_artifact["scaler"]
    scaled = scaler.transform(mlp_frame)
    tensor = torch.tensor(scaled, dtype=torch.float32)
    with torch.no_grad():
        pred_log = mlp_model(tensor).cpu().numpy().flatten()
    prediction = np.expm1(pred_log)
    prediction = np.clip(prediction, 0, 100)
    return pd.Series(prediction, index=mlp_frame.index, name="mlp_prediction")


def predict_all_models(input_df: pd.DataFrame, reference_df: pd.DataFrame, stacking_artifact: dict, mlp_artifact: dict, mlp_model: CoralResidualMLP) -> tuple[pd.DataFrame, list[str]]:
    prepared = prepare_input_frame(input_df, reference_df, stacking_artifact, mlp_artifact)
    stacking_results = predict_stacking_ensemble(stacking_artifact, prepared.stacking)
    mlp_results = predict_custom_residual_mlp(mlp_artifact, mlp_model, prepared.mlp)

    result = prepared.original.copy()
    result = pd.concat([result, stacking_results, mlp_results], axis=1)
    result["hybrid_vs_mlp_gap"] = result["hybrid_prediction"] - result["mlp_prediction"]
    return result, prepared.fill_notes
