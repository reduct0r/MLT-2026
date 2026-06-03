"""
Полный пайплайн типового НИРС: классификация сердечно-сосудистых заболеваний.
Запуск: python scripts/run_pipeline.py
"""
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import FEATURE_LABELS, get_feature_columns, load_heart_disease
from src.metrics_logger import MetricLogger

FIG_DIR = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "models"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 120,
})
sns.set_theme(style="whitegrid")


def save_fig(name: str):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("=== Загрузка данных ===")
    df = load_heart_disease(ROOT / "data")
    feature_cols = get_feature_columns()
    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # --- EDA ---
    print("=== Разведочный анализ ===")
    fig, ax = plt.subplots(figsize=(6, 4))
    df["target_label"].value_counts().plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_title("Распределение целевого класса")
    ax.set_xlabel("Класс")
    ax.set_ylabel("Количество записей")
    save_fig("01_target_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    miss = df[feature_cols].isnull().sum()
    if (miss > 0).any():
        miss[miss > 0].plot(kind="barh", ax=ax, color="#C44E52")
        ax.set_title("Пропуски в признаках")
    else:
        ax.text(0.5, 0.5, "После удаления записей с «?»\nпропусков нет",
                ha="center", va="center", fontsize=12)
        ax.set_title("Пропуски в признаках")
        ax.axis("off")
    save_fig("02_missing_values.png")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, col in zip(axes.flat, ["age", "chol", "thalach", "trestbps", "oldpeak", "thal"]):
        for label, sub in df.groupby("target_label"):
            ax.hist(sub[col], bins=15, alpha=0.6, label=label)
        ax.set_title(FEATURE_LABELS.get(col, col))
        ax.legend(fontsize=8)
    save_fig("03_numeric_histograms.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[feature_cols + ["target"]].copy()
    for c in cat_cols:
        corr[c] = corr[c].astype("category").cat.codes
    sns.heatmap(corr.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Корреляционная матрица признаков")
    save_fig("04_correlation_matrix.png")

    # --- Feature engineering ---
    df["age_group"] = pd.cut(df["age"], bins=[0, 45, 55, 65, 100], labels=["<45", "45-55", "55-65", "65+"])
    df["chol_high"] = (df["chol"] > 240).astype(int)
    df["hr_risk"] = (df["thalach"] < 120).astype(int)

    extended_cat = cat_cols + ["age_group"]
    extended_num = num_cols + ["chol_high", "hr_risk"]
    all_features = extended_num + [c for c in extended_cat if c != "age_group"] + ["age_group"]

    X = df[all_features]
    y = df["target"]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), extended_num),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), extended_cat),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # --- Models ---
    models = {
        "LogReg": LogisticRegression(max_iter=2000, random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVC": SVC(probability=True, random_state=42),
        "Tree": DecisionTreeClassifier(random_state=42),
        "RF": RandomForestClassifier(random_state=42),
        "ET": ExtraTreesClassifier(random_state=42),
        "GB": GradientBoostingClassifier(random_state=42),
    }

    param_grids = {
        "LogReg": {"clf__C": [0.01, 0.1, 1, 10]},
        "KNN": {"clf__n_neighbors": [3, 5, 7, 11, 15]},
        "SVC": {"clf__C": [0.1, 1, 10], "clf__kernel": ["rbf", "linear"]},
        "Tree": {"clf__max_depth": [3, 5, 7, 10, None], "clf__min_samples_leaf": [1, 3, 5]},
        "RF": {"clf__n_estimators": [50, 100, 200], "clf__max_depth": [5, 10, None]},
        "ET": {"clf__n_estimators": [50, 100, 200], "clf__max_depth": [5, 10, None]},
        "GB": {"clf__n_estimators": [50, 100], "clf__learning_rate": [0.05, 0.1, 0.2], "clf__max_depth": [3, 5]},
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    logger = MetricLogger()
    best_models = {}
    grid_results = {}

    def eval_model(pipe, name, stage):
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        logger.add("precision", name, precision_score(y_test, y_pred), stage)
        logger.add("recall", name, recall_score(y_test, y_pred), stage)
        logger.add("f1", name, f1_score(y_test, y_pred), stage)
        logger.add("roc_auc", name, roc_auc_score(y_test, y_proba), stage)
        logger.add("accuracy", name, accuracy_score(y_test, y_pred), stage)
        return pipe

    print("=== Baseline ===")
    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        eval_model(pipe, name, "baseline")

    print("=== GridSearchCV ===")
    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        gs = GridSearchCV(pipe, param_grids[name], cv=cv, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_train, y_train)
        best_models[name] = gs.best_estimator_
        grid_results[name] = {
            "best_params": gs.best_params_,
            "best_cv_roc_auc": float(gs.best_score_),
        }
        eval_model(gs.best_estimator_, name, "tuned")

    # --- Comparison plots ---
    for metric, title, asc in [
        ("roc_auc", "ROC AUC (baseline vs tuned)", False),
        ("f1", "F1-мера (baseline vs tuned)", False),
        ("precision", "Precision (baseline vs tuned)", False),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, stage in zip(axes, ["baseline", "tuned"]):
            labels, vals = logger.get_data_for_metric(metric, stage, ascending=asc)
            ax.barh(range(len(vals)), vals, tick_label=labels)
            ax.set_title(f"{title.split('(')[0].strip()} — {stage}")
            ax.set_xlim(0, 1.05)
        save_fig(f"05_compare_{metric}.png")

    logger.plot("ROC AUC — baseline", "roc_auc", "baseline", save_path=FIG_DIR / "06_roc_auc_baseline.png")
    logger.plot("ROC AUC — tuned", "roc_auc", "tuned", save_path=FIG_DIR / "07_roc_auc_tuned.png")
    logger.plot("F1 — tuned", "f1", "tuned", save_path=FIG_DIR / "08_f1_tuned.png")

    # Best model ROC & CM
    best_name = logger.get_data_for_metric("roc_auc", "tuned", ascending=False)[0][0]
    best_pipe = best_models[best_name]
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_estimator(best_pipe, X_test, y_test, ax=ax)
    ax.set_title(f"ROC-кривая — {best_name} (tuned)")
    save_fig("09_best_roc_curve.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        best_pipe, X_test, y_test, display_labels=["Здоров", "Болезнь"],
        normalize="true", cmap="Blues", ax=ax,
    )
    ax.set_title(f"Матрица ошибок — {best_name}")
    save_fig("10_best_confusion_matrix.png")

    # Hyperparameter influence (RF n_estimators)
    scores, n_list = [], list(range(10, 210, 20))
    for n in n_list:
        p = Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=n, random_state=42)),
        ])
        p.fit(X_train, y_train)
        scores.append(roc_auc_score(y_test, p.predict_proba(X_test)[:, 1]))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n_list, scores, "o-", color="#4C72B0")
    ax.set_xlabel("n_estimators (Random Forest)")
    ax.set_ylabel("ROC AUC на тесте")
    ax.set_title("Влияние числа деревьев на качество")
    save_fig("11_rf_n_estimators.png")

    # Learning curve style (train size vs score)
    sizes = np.linspace(0.2, 1.0, 8)
    train_scores, test_scores = [], []
    for frac in sizes:
        n = int(len(X_train) * frac)
        Xt, yt = X_train.iloc[:n], y_train.iloc[:n]
        p = Pipeline([("prep", preprocessor), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
        p.fit(Xt, yt)
        train_scores.append(roc_auc_score(yt, p.predict_proba(Xt)[:, 1]))
        test_scores.append(roc_auc_score(y_test, p.predict_proba(X_test)[:, 1]))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sizes * len(X_train), train_scores, "o-", label="Train ROC AUC")
    ax.plot(sizes * len(X_train), test_scores, "s-", label="Test ROC AUC")
    ax.set_xlabel("Размер обучающей выборки")
    ax.set_ylabel("ROC AUC")
    ax.legend()
    ax.set_title("Кривые обучения (Random Forest)")
    save_fig("12_learning_curves.png")

    # Save artifacts for Streamlit & report
    import joblib
    joblib.dump(best_pipe, MODEL_DIR / "best_model.joblib")
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")

    sample = X_test.iloc[:1].to_dict(orient="records")[0]
    with open(MODEL_DIR / "demo_sample.json", "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, default=str)

    summary = {
        "dataset_rows": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "best_model": best_name,
        "grid_results": grid_results,
        "metrics": logger.df.to_dict(orient="records"),
    }
    with open(FIG_DIR / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    pd.DataFrame(logger.df).to_csv(FIG_DIR / "metrics_table.csv", index=False)
    print(f"Готово. Лучшая модель: {best_name}. Графики: {FIG_DIR}")


if __name__ == "__main__":
    main()
