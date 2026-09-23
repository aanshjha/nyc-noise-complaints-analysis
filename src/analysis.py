"""Reusable analysis code for the NYC 311 noise-complaint midterm."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "nyc_311_noise_2024-06-30_to_2024-07-06.csv"
VALID_BOROUGHS = {"BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"}


@dataclass(frozen=True)
class CleaningAudit:
    raw_rows: int
    duplicate_rows: int
    invalid_date_rows: int
    negative_response_rows: int
    analysis_rows: int


@dataclass(frozen=True)
class ModelResult:
    pipeline: Pipeline
    metrics: dict[str, float]
    confusion_matrix: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    coefficients: pd.DataFrame


def load_raw_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the committed snapshot without coercing ZIP codes to numbers."""
    return pd.read_csv(path, dtype={"incident_zip": "string", "unique_key": "string"})


def _snake_case(columns: pd.Index) -> pd.Index:
    return (
        columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )


def clean_complaints(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningAudit]:
    """Validate, de-duplicate, and derive analysis-ready complaint fields."""
    data = raw.copy()
    data.columns = _snake_case(data.columns)

    required = {
        "unique_key",
        "created_date",
        "closed_date",
        "complaint_type",
        "borough",
        "incident_zip",
        "open_data_channel_type",
        "latitude",
        "longitude",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    duplicate_rows = int(data.duplicated(subset="unique_key", keep="first").sum())
    data = data.drop_duplicates(subset="unique_key", keep="first").copy()

    data["created_date"] = pd.to_datetime(data["created_date"], errors="coerce")
    data["closed_date"] = pd.to_datetime(data["closed_date"], errors="coerce")
    invalid_dates = data[["created_date", "closed_date"]].isna().any(axis=1)
    invalid_date_rows = int(invalid_dates.sum())
    data = data.loc[~invalid_dates].copy()

    data["response_time_hours"] = (
        data["closed_date"] - data["created_date"]
    ).dt.total_seconds() / 3600
    negative = data["response_time_hours"] < 0
    negative_response_rows = int(negative.sum())
    data = data.loc[~negative].copy()

    data["borough"] = data["borough"].str.upper().where(
        data["borough"].str.upper().isin(VALID_BOROUGHS)
    )
    data["incident_zip"] = data["incident_zip"].str.extract(r"(\d{5})", expand=False)
    data["latitude"] = pd.to_numeric(data["latitude"], errors="coerce")
    data["longitude"] = pd.to_numeric(data["longitude"], errors="coerce")
    data["hour"] = data["created_date"].dt.hour
    data["day_of_week"] = data["created_date"].dt.day_name()
    data["is_weekend"] = data["created_date"].dt.dayofweek.ge(5).astype(int)
    data["time_of_day"] = np.select(
        [data["hour"].between(7, 18), data["hour"].between(19, 22)],
        ["Daytime", "Evening"],
        default="Overnight",
    )
    data["over_two_hours"] = data["response_time_hours"].ge(2).astype(int)

    audit = CleaningAudit(
        raw_rows=len(raw),
        duplicate_rows=duplicate_rows,
        invalid_date_rows=invalid_date_rows,
        negative_response_rows=negative_response_rows,
        analysis_rows=len(data),
    )
    return data.reset_index(drop=True), audit


def response_summary(data: pd.DataFrame, group: str) -> pd.DataFrame:
    """Summarize closure times and two-hour rates for one grouping variable."""
    summary = (
        data.dropna(subset=[group])
        .groupby(group, observed=True)
        .agg(
            n=("unique_key", "size"),
            median_hours=("response_time_hours", "median"),
            mean_hours=("response_time_hours", "mean"),
            over_two_hour_rate=("over_two_hours", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values("over_two_hour_rate", ascending=False).reset_index(drop=True)


def _cramers_v(table: pd.DataFrame, chi2: float) -> float:
    n = table.to_numpy().sum()
    denominator = n * min(table.shape[0] - 1, table.shape[1] - 1)
    return float(np.sqrt(chi2 / denominator)) if denominator else np.nan


def test_group_differences(data: pd.DataFrame) -> pd.DataFrame:
    """Run nonparametric response-time and binary-outcome association tests."""
    rows: list[dict[str, float | str]] = []
    for group in ["borough", "complaint_type", "day_of_week"]:
        subset = data.dropna(subset=[group, "response_time_hours"])
        samples = [values.to_numpy() for _, values in subset.groupby(group)["response_time_hours"]]
        statistic, p_value = kruskal(*samples)
        rows.append(
            {
                "variable": group,
                "test": "Kruskal-Wallis",
                "statistic": statistic,
                "p_value": p_value,
                "effect_size": np.nan,
            }
        )

        table = pd.crosstab(subset[group], subset["over_two_hours"])
        chi2, p_value, _, _ = chi2_contingency(table)
        rows.append(
            {
                "variable": group,
                "test": "Chi-square",
                "statistic": chi2,
                "p_value": p_value,
                "effect_size": _cramers_v(table, chi2),
            }
        )
    return pd.DataFrame(rows)


def fit_response_model(data: pd.DataFrame, random_state: int = 42) -> ModelResult:
    """Fit and evaluate a leakage-safe logistic regression baseline."""
    categorical = [
        "borough",
        "complaint_type",
        "open_data_channel_type",
        "day_of_week",
        "time_of_day",
    ]
    numeric = ["hour", "is_weekend", "latitude", "longitude"]
    features = categorical + numeric

    X = data[features]
    y = data["over_two_hours"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    preprocessing = ColumnTransformer(
        [("categorical", categorical_pipeline, categorical), ("numeric", numeric_pipeline, numeric)]
    )
    pipeline = Pipeline(
        [
            ("preprocess", preprocessing),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    names = [name.replace("categorical__", "").replace("numeric__", "") for name in names]
    coefficients = pd.DataFrame(
        {"feature": names, "coefficient": pipeline.named_steps["model"].coef_[0]}
    )

    return ModelResult(
        pipeline=pipeline,
        metrics=metrics,
        confusion_matrix=confusion_matrix(y_test, predictions),
        fpr=fpr,
        tpr=tpr,
        coefficients=coefficients,
    )
