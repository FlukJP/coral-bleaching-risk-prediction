# Project: Ocean Climate & Coral Bleaching Analysis

## 1. Project Overview

This project is a Streamlit web application for exploring and analyzing ocean climate, marine life, and coral bleaching data.

The application is built using Python and Streamlit.

### Main Objectives

- Explore global coral bleaching and environmental data.
- Analyze ocean climate conditions and marine life.
- Visualize patterns, trends, and relationships in the datasets.
- Present data analysis results through interactive charts and dashboards.
- Provide an accessible interface for understanding ocean environmental changes.

### Technology Stack

- Programming Language: Python
- Web Framework: Streamlit
- Data Processing: pandas, NumPy
- Data Visualization: Follow the libraries already used in the repository.
- Development Environment: Python virtual environment

Do not introduce additional frameworks or libraries unless they are necessary for the requested task.

---

## 2. Data Sources

The application uses the following datasets.

### Dataset 1: Global Bleaching Environmental Data

File: `global_bleaching_environmental.csv`

Source:
https://www.bco-dmo.org/dataset/773466

Dataset description:

Bleaching and environmental data for global coral reef sites from 1980–2020.

Project information:

- Principal Investigator: Robert van Woesik (Florida Institute of Technology)
- Co-Principal Investigator: Deron Burkepile (University of California-Santa Barbara)
- Data Manager: Chelsey Kratochwill (Florida Institute of Technology)
- BCO-DMO Data Manager: Shannon Rauch (Woods Hole Oceanographic Institution)

### Dataset 2: Ocean Climate and Marine Life Dataset

File: `realistic_ocean_climate_dataset.csv`

Source:
https://www.kaggle.com/datasets/atharvasoundankar/shifting-seas-ocean-climate-and-marine-life-dataset/data

Dataset description:

Shifting Seas - Ocean Climate & Marine Life Dataset.

Collaborator: Atharva Soundankar

### Data Handling Rules

- Always inspect the dataset columns and data types before using them.
- Do not assume column names or data formats without checking the actual CSV files.
- Handle missing values appropriately.
- Check for duplicate records when relevant.
- Validate data before performing calculations or visualizations.
- Preserve the original datasets.
- Do not overwrite or modify original CSV files during data preprocessing.
- Clearly distinguish between actual observations and derived or calculated values.
- Do not invent missing data or fabricate analysis results.
- Preserve dataset attribution when displaying or documenting data sources.

---

## 3. Application Execution

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run streamlit_app.py
```

### Development Rules

- Use the existing Python environment and dependency configuration.
- Check `requirements.txt` before installing new packages.
- Do not remove or change dependencies without understanding their impact.
- Keep the application executable using the documented Streamlit command.
- Verify that the application starts successfully after code changes.

---

## 4. Repository Exploration

Before modifying the project:

1. Inspect the repository structure.
2. Read `streamlit_app.py`.
3. Read `requirements.txt`.
4. Identify all Python modules used by the application.
5. Locate and inspect the available CSV datasets.
6. Understand the existing application workflow.
7. Identify dependencies between components.

Do not assume that the repository structure matches a generic Streamlit template.

Always follow the actual structure and conventions found in the repository.

---

## 5. Coding Conventions

### Python

- Follow PEP 8 coding conventions.
- Use meaningful variable and function names.
- Keep functions focused on a single responsibility.
- Avoid unnecessary code duplication.
- Use appropriate exception handling.
- Add comments or docstrings for complex logic.
- Avoid using `eval()` for processing user input.
- Avoid using global variables unnecessarily.

### Streamlit

- Use Streamlit components consistently with the existing application.
- Keep the user interface simple and understandable.
- Use appropriate layouts, tabs, columns, and containers.
- Provide clear labels for charts, metrics, and input controls.
- Use `st.cache_data` when appropriate for expensive data-loading or preprocessing operations.
- Avoid unnecessary application reruns.
- Handle empty datasets and invalid user selections gracefully.

### Data Processing

- Use pandas and NumPy for data processing when appropriate.
- Separate data loading, preprocessing, analysis, and visualization where practical.
- Avoid modifying the original dataset.
- Ensure calculations use appropriate data types.
- Handle missing and invalid values before analysis.
- Make transformations reproducible.

---

## 6. Data Analysis and Visualization Rules

The application focuses on ocean climate and coral bleaching analysis.

When implementing analytical features:

- Use only columns and variables that actually exist in the datasets.
- Explain the meaning of important metrics.
- Use appropriate statistical methods for each analytical task.
- Clearly identify the time period and geographical scope of the data.
- Ensure that charts accurately represent the underlying data.
- Include chart titles, axis labels, units, and legends where applicable.
- Avoid misleading visualizations.
- Do not interpret correlation as causation.
- Distinguish between historical observations and predictions.

When combining datasets:

- Inspect the schema of both datasets first.
- Identify appropriate common variables or join keys.
- Check differences in units, formats, and measurement periods.
- Avoid joining datasets without a valid analytical reason.
- Document assumptions and limitations.

---

## 7. Error Handling

The application should handle common problems gracefully.

Examples:

- Missing CSV files.
- Incorrect file paths.
- Empty datasets.
- Missing columns.
- Invalid data types.
- Missing values.
- Invalid user inputs.
- Data loading failures.
- Visualization errors.

Do not silently ignore important errors.

Display understandable error messages using Streamlit components where appropriate.

Avoid exposing sensitive information or internal system details to application users.

---

## 8. Project Safety Rules

The following rules must be followed:

- Do not delete existing files unless explicitly requested.
- Do not overwrite original datasets.
- Do not modify dataset source information without verification.
- Do not change the application's main entry point without a clear reason.
- Do not modify `requirements.txt` unnecessarily.
- Do not introduce unnecessary dependencies.
- Do not expose API keys, passwords, or credentials.
- Do not commit `.env` files or other sensitive information.
- Do not perform unrelated refactoring.
- Do not remove existing functionality without authorization.

Before making significant changes, inspect the impact on the existing application.

---

## 9. Agent Workflow

Follow this workflow for every development task.

### Step 1: Understand

- Read the user's request carefully.
- Inspect the relevant source code.
- Understand the existing implementation.
- Identify affected files and dependencies.
- Check whether the requested feature already exists.

### Step 2: Plan

- Explain the proposed implementation.
- Identify files that need to be created or modified.
- Identify potential risks and compatibility issues.
- Keep the implementation scope limited to the requested task.

### Step 3: Implement

- Make small and focused changes.
- Follow existing project conventions.
- Reuse existing functions and components when possible.
- Preserve existing functionality.
- Avoid unrelated modifications.

### Step 4: Verify

After implementation:

1. Check Python syntax.
2. Check imports and dependencies.
3. Verify dataset loading.
4. Run relevant tests if available.
5. Run the Streamlit application.
6. Verify the modified features.
7. Review the final Git diff for unintended changes.

Never claim that a test or verification passed unless it was actually executed.

If verification cannot be completed, clearly explain the limitation.

### Step 5: Report

After completing the task, provide:

- Summary of implemented changes.
- List of modified and created files.
- Explanation of important implementation decisions.
- Tests and verification results.
- Known issues or remaining limitations.

---

## 10. Definition of Done

A task is considered complete only when:

- The requested feature or fix is implemented.
- The application remains compatible with Streamlit.
- Existing functionality is preserved.
- Dataset handling is correct.
- No original dataset has been unintentionally modified.
- Relevant tests and checks have been executed.
- No unrelated files have been changed.
- Remaining limitations are documented.

---

## 11. Important Instructions

Prioritize the following principles:

1. Understand before modifying.
2. Preserve existing working functionality.
3. Protect original datasets.
4. Make minimal and maintainable changes.
5. Verify every important implementation.
6. Report actual results honestly.

The goal is to improve the application without introducing regressions or unnecessary complexity.