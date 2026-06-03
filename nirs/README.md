# НИРС: Прогнозирование сердечно-сосудистых заболеваний

Типовое исследование по дисциплине «Технологии машинного обучения».

**Задача:** бинарная классификация по набору [Heart Disease (Cleveland)](https://archive.ics.uci.edu/ml/datasets/Heart+Disease).

## Структура

- `data/` — исходные и обработанные данные
- `scripts/run_pipeline.py` — полный ML-пайплайн (EDA, модели, GridSearch, графики)
- `scripts/generate_report.py` — генерация отчёта Word
- `app/streamlit_app.py` — веб-демонстрация с настройкой `n_estimators`
- `reports/figures/` — графики для отчёта
- `reports/Отчет_НИРС.docx` — отчёт (без титульного листа)

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

```bash
python scripts/run_pipeline.py
python scripts/generate_report.py
streamlit run app/streamlit_app.py
```

Титульный лист и бланк задания вставляются в Word вручную по шаблону вуза.
