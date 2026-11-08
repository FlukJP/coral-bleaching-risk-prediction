# Coral Bleaching Risk Prediction

## 1. Project Overview

This repository contains a Python and Streamlit application for predicting coral bleaching severity from processed environmental and geographic features. The app presents a dashboard and a single-record prediction interface backed by two trained models: a Stacking Ensemble and a Custom Residual MLP.

The application is an inference interface, not a general-purpose ocean data exploration dashboard. Model training and dataset preparation are documented in Jupyter notebooks under `training/`.

### Main components

- `streamlit_app.py`: application entry point and model/data status dashboard.
- `pages/1_Coral_Bleaching_AI_Studio.py`: prediction form, model results, and CSV downloads.
- `model_inference.py`: feature preparation, artifact loading, and predictions for both models.
- `training/`: preprocessing and model-training notebooks.
- `data/`: source datasets and the derived model-ready dataset.
- `model/`: serialized model artifacts used by the app.

## 2. Repository Structure

```text
.
├── .devcontainer/
│   └── devcontainer.json
├── .github/
│   └── CODEOWNERS
├── .gitignore
├── data/
│   ├── global_bleaching_environmental.csv
│   ├── realistic_ocean_climate_dataset.csv
│   └── bleaching_model_ready.csv
├── model/
│   ├── custom_residual_mlp/
│   │   └── custom_residual_mlp_artifact.pt
│   └── stacking_ensemble/
│       └── Stacking_Ensemble_model.joblib
├── pages/
│   └── 1_Coral_Bleaching_AI_Studio.py
├── training/
│   ├── Clean_data.ipynb
│   ├── Custom_Residual_MLP.ipynb
│   └── Stacking_Ensemble.ipynb
├── AGENTS.md
├── claude-progress.md
├── LICENSE
├── README.md
├── model_inference.py
├── requirements.txt
└── streamlit_app.py
```

`claude-progress.md` is an empty auxiliary file in the audited tree. Do not assume it is part of the application workflow.

### Component relationships

1. `training/Clean_data.ipynb` reads the two source CSVs and writes `data/bleaching_model_ready.csv`.
2. `training/Stacking_Ensemble.ipynb` and `training/Custom_Residual_MLP.ipynb` train models using the model-ready CSV and export their artifacts under `model/`.
3. `streamlit_app.py` imports artifact/data paths from `model_inference.py` and links to the prediction page.
4. The prediction page uses `model_inference.py` to load the reference feature schema and model artifacts, prepare submitted values, generate predictions, and offer downloadable results.

## 3. Technology and Execution

The project uses Python, Streamlit, pandas, NumPy, scikit-learn, PyTorch, joblib, XGBoost, LightGBM, Optuna, SHAP, and Matplotlib across the application and training notebooks. The dependencies are listed without version pins in `requirements.txt`; check that file before changing or installing dependencies.

Run the app from the repository root:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Do not install additional libraries or change dependency versions unless the task requires it. Preserve the documented Streamlit entry point.

## 4. Application Workflow

1. Streamlit starts `streamlit_app.py`, which displays the application dashboard and checks whether the expected model artifacts and model-ready CSV exist.
2. The prediction page loads `data/bleaching_model_ready.csv` as a reference schema, dropping `Percent_Bleaching` to obtain the 61 inference features. Model resources are cached.
3. The form accepts one record at a time. Its initial values are representative medians/modes from the reference data, not measurements from a real site.
4. `model_inference.py` validates required features, coerces boolean/numeric values, fills invalid or missing values from reference data, and creates the model-specific feature frames.
5. Both models produce predictions. The page displays the severity estimates and Stacking risk outputs and supports CSV downloads.

The prediction is a model estimate, not an observed bleaching measurement or a guarantee of future conditions. Do not describe representative form defaults as real observations.

## 5. Machine Learning Architecture

The prediction target is `Percent_Bleaching`, expressed as a percentage. Both training pipelines transform the target with `log1p`; inference converts predictions back with `expm1` and clips severity estimates to `[0, 100]`.

### Stacking Ensemble

- Training workflow: `training/Stacking_Ensemble.ipynb`.
- Artifact: `model/stacking_ensemble/Stacking_Ensemble_model.joblib`.
- Severity model: scikit-learn `StackingRegressor` combining Random Forest, XGBoost, and LightGBM regressors, with a standardized Ridge final estimator.
- Risk workflow: an XGBoost classifier predicts whether bleaching is present (`Percent_Bleaching > 0`); a separate XGBoost regressor estimates severity for cases classified above the stored risk threshold.
- Final hybrid severity blends the stacking severity and two-stage severity using the artifact's stored threshold and blend weight. Inference also returns the risk probability, risk flag, and component predictions.
- The notebook's audited evaluation configuration uses grouped spatial validation (`GroupKFold`, four folds). Do not report model accuracy without checking the current artifact's evaluation results.

### Custom Residual MLP

- Training workflow: `training/Custom_Residual_MLP.ipynb`.
- Artifact: `model/custom_residual_mlp/custom_residual_mlp_artifact.pt`.
- The PyTorch network applies tabular attention, projects the input to a 128-unit hidden representation, processes three residual blocks, and uses an output head for the log-transformed severity.
- The artifact contains the model state, configuration, feature names, and a fitted `StandardScaler`; inference must use the matching scaler and feature order.
- The notebook's audited evaluation uses spatial `GroupKFold` with three folds.

## 6. Input Schema and Target

`data/bleaching_model_ready.csv` contains 62 columns: the `Percent_Bleaching` target plus the following 61 ordered model inputs. The target must not be included in inference input.

```text
Latitude_Degrees
Longitude_Degrees
Distance_to_Shore
Turbidity
Cyclone_Frequency
Date_Day
Date_Month
Date_Year
Depth_m
ClimSST
Temperature_Mean
Temperature_Minimum
Temperature_Maximum
Temperature_Kelvin_Standard_Deviation
Windspeed
SSTA
SSTA_Standard_Deviation
SSTA_Mean
SSTA_Minimum
SSTA_Maximum
SSTA_Frequency
SSTA_Frequency_Standard_Deviation
SSTA_FrequencyMax
SSTA_FrequencyMean
SSTA_DHW
SSTA_DHW_Standard_Deviation
SSTA_DHWMax
SSTA_DHWMean
TSA
TSA_Standard_Deviation
TSA_Minimum
TSA_Maximum
TSA_Mean
TSA_Frequency
TSA_Frequency_Standard_Deviation
TSA_FrequencyMax
TSA_FrequencyMean
TSA_DHW
TSA_DHW_Standard_Deviation
TSA_DHWMax
TSA_DHWMean
Temperature_C
Temp_Range_C
Is_Tropical
Aux_Marine_Heatwave
Aux_pH_Level
Ocean_Name_Atlantic
Ocean_Name_Indian
Ocean_Name_Pacific
Ocean_Name_Red Sea
Realm_Name_Eastern Indo-Pacific
Realm_Name_Temperate Australasia
Realm_Name_Temperate Northern Atlantic
Realm_Name_Temperate Northern Pacific
Realm_Name_Tropical Atlantic
Realm_Name_Tropical Eastern Pacific
Realm_Name_Western Indo-Pacific
Exposure_Sheltered
Exposure_Sometimes
Substrate_Name_Hard Coral
Substrate_Name_Nutrient Indicator Algae
```

Treat names, data types, and order as a model contract. Inspect the current CSV and artifact metadata before changing or documenting a schema; preprocessing changes may alter one-hot encoded columns.

## 7. Data Sources, Integrity, and Matching Rules

### Source datasets

1. `data/global_bleaching_environmental.csv` — BCO-DMO dataset 773466, bleaching and environmental observations for global reef sites (1980–2020). Preserve the source attribution: PI Robert van Woesik; Co-PI Deron Burkepile; data managers Chelsey Kratochwill and Shannon Rauch.
2. `data/realistic_ocean_climate_dataset.csv` — “Shifting Seas - Ocean Climate & Marine Life Dataset,” attributed to Atharva Soundankar on Kaggle. Its audited schema includes dates, coordinates, SST, pH, bleaching severity, species observed, and marine-heatwave status.
3. `data/bleaching_model_ready.csv` — derived training/reference data, not an original source dataset.

### Integrity requirements

- Never overwrite or modify either source CSV during cleaning, training, or validation. Write derived data only to the designated model-ready output or another explicitly approved derived-data path.
- Inspect the actual columns, types, units, date coverage, missing values, and duplicates before using or combining data. Do not assume schema based only on documentation.
- Convert documented missing-value markers deliberately; do not silently treat missing or invalid values as observations.
- Investigate duplicate rows before removing them. The last repository audit found 155 exact duplicate rows in the model-ready CSV; verify against the current file before relying on that count.
- Keep observations distinct from engineered or imputed values, and document imputation and feature-engineering assumptions.
- Avoid interpreting correlation or model output as causation.

### Temporal and geographical matching

`training/Clean_data.ipynb` augments global bleaching observations with marine-heatwave and pH features from the second dataset. Matching must use both geography and time:

- Require valid coordinates and dates for a match.
- Consider source records within 500 km and within 30 days of the observation date.
- Among eligible records, use the geographically nearest one.
- If no source record meets both limits, leave the auxiliary values missing for the notebook's existing regional-then-global median imputation. Do not encode “no match” as an observed no-heatwave value.
- Do not widen these limits, remove the date condition, or change unmatched-value handling without documenting the rationale and checking model implications.

The last audit found only 801 spatial-and-temporal matches in 41,361 global observations. Consequently, the auxiliary fields may rely heavily on imputation; evaluate their coverage and utility rather than assuming they are measured for each row.

## 8. Model Artifact and Inference Compatibility

- Do not edit, replace, or delete `.joblib` or `.pt` artifacts unless explicitly requested.
- Preserve the 61 feature names and their expected order, the preprocessing transformations, and the scaler/configuration embedded or referenced by each artifact.
- Before changing preprocessing or the model-ready schema, inspect both training notebooks, `model_inference.py`, and artifact metadata. A change to features, encoding, imputation, or ordering can invalidate inference compatibility.
- Re-run preprocessing and retrain/export both models when a changed training schema or feature distribution requires it; do not assume updating a notebook automatically updates the derived CSV or model artifacts.
- Check package compatibility when loading serialized scikit-learn/joblib or PyTorch artifacts. Do not silently substitute an incompatible artifact or default.
- Validate prediction shape, finiteness, and expected range before displaying results. Distinguish severity estimates from risk probability and binary risk flags.
- Do not claim accuracy or production readiness without current, relevant evaluation evidence.

## 9. Known Repository Risks

These are findings from the last repository audit; recheck current files before relying on counts or state:

- The audited `bleaching_model_ready.csv` had 34,515 rows, 62 columns, and 155 exact duplicate rows.
- The second dataset had 500 rows and 150 missing `Bleaching Severity` values. Its coverage and schema differ from the historical bleaching dataset.
- The date-aware join has limited coverage (801 matched rows in the audited data); unmatched auxiliary values are median-imputed, which may weaken their observational meaning.
- The model-ready CSV and shipped model artifacts were not regenerated/retrained as part of the notebook-only date-matching change. They may therefore reflect the earlier spatial-only join until the preprocessing and model-training workflow is rerun and artifacts are deliberately updated.
- The dashboard's “Fresh” data status checks file existence rather than the age or provenance of the data.
- The prediction UI catches broad exceptions and displays exception text; avoid exposing internal details when changing error handling.
- `.devcontainer/devcontainer.json` runs Streamlit with CORS and XSRF protection disabled. Do not use that configuration for an untrusted or public deployment without restoring appropriate protections.
- `README.md` still contains blank-app-template wording, dependencies are unpinned, and no automated test suite was present in the last audit.

Treat these as risks to verify and address when in scope, not as permission for unrelated changes.

## 10. Coding and Application Guidelines

### Python

- Follow PEP 8 and existing naming, formatting, and module conventions.
- Use meaningful names and focused functions; avoid unnecessary duplication and global state.
- Keep types and input validation explicit; avoid `eval()` for user input.
- Add comments/docstrings only where they clarify non-obvious behavior.
- Handle errors explicitly. Do not silently ignore failures, return success-shaped fallbacks, or expose sensitive/internal details to users.

### Streamlit

- Follow the existing multipage layout and component patterns.
- Keep labels and explanations clear, especially for units, defaults, predictions, probabilities, and imputed values.
- Use caching for expensive data/artifact loading where it matches existing patterns.
- Handle missing files, empty data, invalid input, and incompatible artifacts with useful user-facing messages and appropriate internal diagnostics.

### Data and model work

- Keep loading, preprocessing, training, and inference behavior consistent and reproducible.
- Preserve source datasets and source attribution.
- Validate input and output types, missing values, duplicates, date and geographic scope, and metric interpretation.
- Distinguish historical observations, derived values, imputations, and model predictions.

## 11. Development Workflow and Verification

Before making a change:

1. Read this file and inspect the repository structure.
2. Read the relevant application module, notebook, data schema, and dependency configuration.
3. Trace the affected workflow and model/data dependencies.
4. Identify the smallest in-scope change and any compatibility risks.

When implementing:

- Make precise, focused changes and preserve existing behavior outside the requested scope.
- Do not modify unrelated files, datasets, or artifacts.
- Do not change the app entry point or dependency manifest without a clear requirement.

After implementing:

1. Run the smallest relevant syntax, test, or validation command available.
2. Verify dataset loading and schema if data handling changed.
3. Verify model/artifact compatibility if inference or preprocessing changed.
4. Start the Streamlit app when the change affects app execution and the environment permits it.
5. Review the final Git diff for unintended changes.
6. Report only checks actually performed and state any limitations.

Do not install dependencies unless a manifest was intentionally changed or a required validation fails because a dependency is missing. Documentation-only changes do not require application tests unless documentation checks exist.

## 12. Safety and Scope

- Do not delete files, overwrite original datasets, expose credentials, or commit secrets.
- Do not alter source attribution without verification.
- Do not retrain models, replace artifacts, or remove functionality without authorization.
- Do not introduce frameworks or dependencies without need.
- Do not perform unrelated refactoring.
- Before significant or potentially breaking changes, explain the impact and follow the user's requested approval workflow.

## 13. Reporting

Summarize the change, list modified files, explain important decisions, and report actual test/verification results and remaining limitations. Use repository-relative file paths in project documentation.
