"""Загрузка и подготовка датасета UCI Heart Disease (Cleveland)."""
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

FEATURE_LABELS = {
    "age": "Возраст (лет)",
    "sex": "Пол (1 — мужской)",
    "cp": "Тип боли в груди",
    "trestbps": "Артериальное давление в покое",
    "chol": "Холестерин (мг/дл)",
    "fbs": "Сахар крови > 120 мг/дл",
    "restecg": "ЭКГ в покое",
    "thalach": "Макс. ЧСС на нагрузке",
    "exang": "Стенокардия при нагрузке",
    "oldpeak": "Депрессия ST",
    "slope": "Наклон ST",
    "ca": "Число сосудов (флюорография)",
    "thal": "Талассемия",
}


def download_raw(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / "heart_cleveland.raw"
    if not raw_path.exists():
        urllib.request.urlretrieve(DATA_URL, raw_path)
    return raw_path


def load_heart_disease(data_dir: str | Path = "data") -> pd.DataFrame:
    data_dir = Path(data_dir)
    raw_path = download_raw(data_dir)

    df = pd.read_csv(raw_path, names=COLUMN_NAMES, na_values="?")
    df = df.dropna().copy()
    df["target"] = (df["target"] > 0).astype(int)
    df["target_label"] = df["target"].map({0: "Здоров", 1: "Болезнь"})

    processed_path = data_dir / "heart_disease.csv"
    df.to_csv(processed_path, index=False)
    return df


def get_feature_columns() -> list[str]:
    return [c for c in COLUMN_NAMES if c != "target"]
