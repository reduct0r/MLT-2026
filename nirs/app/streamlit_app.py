"""
Веб-демонстрация модели: Random Forest с настройкой n_estimators.
Запуск: streamlit run app/streamlit_app.py
"""
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_heart_disease

st.set_page_config(page_title="НИРС: прогноз ССЗ", layout="wide")
st.title("Прогнозирование сердечно-сосудистых заболеваний")
st.markdown(
    "Интерактивная демонстрация модели **Random Forest**. "
    "При изменении гиперпараметра `n_estimators` модель переобучается на лету."
)

@st.cache_data
def load_data():
    return load_heart_disease(ROOT / "data")


@st.cache_resource
def prepare_splits():
    df = load_data()
    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    df["age_group"] = pd.cut(df["age"], bins=[0, 45, 55, 65, 100], labels=["<45", "45-55", "55-65", "65+"])
    df["chol_high"] = (df["chol"] > 240).astype(int)
    df["hr_risk"] = (df["thalach"] < 120).astype(int)
    extended_cat = cat_cols + ["age_group"]
    extended_num = num_cols + ["chol_high", "hr_risk"]
    features = extended_num + [c for c in extended_cat if c != "age_group"] + ["age_group"]
    X = df[features]
    y = df["target"]
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


def build_preprocessor(extended_num, extended_cat):
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), extended_num),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), extended_cat),
    ])


df = load_data()
X_train, X_test, y_train, y_test = prepare_splits()

demo_path = ROOT / "models" / "demo_sample.json"
if demo_path.exists():
    with open(demo_path, encoding="utf-8") as f:
        default_patient = json.load(f)
else:
    default_patient = X_test.iloc[0].to_dict()

col_params, col_data = st.columns([1, 2])

with col_params:
    st.subheader("Гиперпараметры модели")
    n_estimators = st.slider(
        "n_estimators (число деревьев в лесу)",
        min_value=10, max_value=300, value=100, step=10,
    )
    max_depth = st.selectbox(
        "max_depth",
        options=[None, 3, 5, 7, 10, 15],
        index=2,
        format_func=lambda x: "без ограничения" if x is None else str(x),
    )

with col_data:
    st.subheader("Данные пациента")
    patient = {}
    for col in ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal"]:
        if col in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]:
            opts = sorted(df[col].dropna().unique().astype(int))
            default_val = int(default_patient.get(col, opts[0]))
            idx = opts.index(default_val) if default_val in opts else 0
            patient[col] = st.selectbox(col, opts, index=idx)
        else:
            patient[col] = st.number_input(
                col,
                float(df[col].min()), float(df[col].max()),
                float(default_patient.get(col, df[col].median())),
            )

# Train model on slider change
cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
extended_cat = cat_cols + ["age_group"]
extended_num = num_cols + ["chol_high", "hr_risk"]

train_df = X_train.copy()
train_df["age_group"] = pd.cut(train_df["age"], bins=[0, 45, 55, 65, 100], labels=["<45", "45-55", "55-65", "65+"])
train_df["chol_high"] = (train_df["chol"] > 240).astype(int)
train_df["hr_risk"] = (train_df["thalach"] < 120).astype(int)

preprocessor = build_preprocessor(extended_num, extended_cat)
pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )),
])
pipe.fit(train_df[extended_num + extended_cat], y_train)

row = pd.DataFrame([patient])
row["age_group"] = pd.cut(row["age"], bins=[0, 45, 55, 65, 100], labels=["<45", "45-55", "55-65", "65+"])
row["chol_high"] = (row["chol"] > 240).astype(int)
row["hr_risk"] = (row["thalach"] < 120).astype(int)

proba = pipe.predict_proba(row[extended_num + extended_cat])[0]
pred = int(proba[1] >= 0.5)

st.subheader("Результат")
m1, m2, m3 = st.columns(3)
y_pred_all = pipe.predict(X_test)
y_proba_all = pipe.predict_proba(X_test)[:, 1]
m1.metric("Вероятность заболевания", f"{proba[1]:.1%}")
m2.metric("Прогноз", "Болезнь" if pred == 1 else "Здоров")
m3.metric("ROC AUC (тест)", f"{roc_auc_score(y_test, y_proba_all):.3f}")

st.progress(float(proba[1]), text="Вероятность положительного класса")
st.caption(
    f"Accuracy на тесте: {accuracy_score(y_test, y_pred_all):.3f} | "
    f"F1: {f1_score(y_test, y_pred_all):.3f}"
)
