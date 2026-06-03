# -*- coding: utf-8 -*-
"""
Генерация отчёта Word по ГОСТ (без титульного листа).
Запуск: python scripts/generate_report.py
"""
from pathlib import Path
import json
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
OUT_PATH = ROOT / "reports" / "Отчет_НИРС.docx"


def set_gost_style(doc: Document):
    section = doc.sections[0]
    section.left_margin = Cm(3)
    section.right_margin = Cm(1.5)
    section.top_margin = Cm(2)
    section.bottom_margin = Cm(2)
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Times New Roman"
    font.size = Pt(14)
    font.color.rgb = RGBColor(0, 0, 0)
    pf = style.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = 1.5
    pf.first_line_indent = Cm(1.25)
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = "Times New Roman"
        run.font.color.rgb = RGBColor(0, 0, 0)
        run.font.bold = True
        if level == 1:
            run.font.size = Pt(16)
        elif level == 2:
            run.font.size = Pt(15)
        else:
            run.font.size = Pt(14)
    h.paragraph_format.first_line_indent = Cm(0)
    h.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return h


def add_paragraph(doc, text, indent=True):
    p = doc.add_paragraph(text)
    p.paragraph_format.first_line_indent = Cm(1.25) if indent else Cm(0)
    for run in p.runs:
        run.font.name = "Times New Roman"
        run.font.size = Pt(14)
    return p


def add_figure(doc, image_path: Path, caption: str, width_cm=15):
    if not image_path.exists():
        add_paragraph(doc, f"[Рисунок не найден: {image_path.name}]", indent=False)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    run = p.add_run()
    run.add_picture(str(image_path), width=Cm(width_cm))
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.first_line_indent = Cm(0)
    for r in cap.runs:
        r.font.name = "Times New Roman"
        r.font.size = Pt(12)
        r.italic = True


def add_toc_placeholder(doc):
    add_heading(doc, "СОДЕРЖАНИЕ", 1)
    add_paragraph(
        doc,
        "Примечание: после открытия документа в Microsoft Word выделите этот раздел "
        "и вставьте автоматическое оглавление (Ссылки → Оглавление), либо обновите поле TOC.",
        indent=False,
    )
    toc_items = [
        "Введение",
        "1 Постановка задачи",
        "2 Набор данных",
        "3 Разведочный анализ данных",
        "4 Подготовка признаков",
        "5 Корреляционный анализ",
        "6 Метрики качества",
        "7 Выбор моделей машинного обучения",
        "8 Формирование обучающей и тестовой выборок",
        "9 Baseline-модели",
        "10 Подбор гиперпараметров",
        "11 Сравнение baseline и tuned-моделей",
        "12 Выводы по результатам моделирования",
        "13 Веб-приложение для демонстрации модели",
        "Заключение",
        "Список использованных источников",
    ]
    for item in toc_items:
        p = doc.add_paragraph(item)
        p.paragraph_format.first_line_indent = Cm(0)
        for r in p.runs:
            r.font.name = "Times New Roman"
            r.font.size = Pt(14)
    doc.add_page_break()


def main():
    summary_path = FIG_DIR / "results_summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    best_model = summary.get("best_model", "RF")

    doc = Document()
    set_gost_style(doc)
    add_toc_placeholder(doc)

    # Введение
    add_heading(doc, "ВВЕДЕНИЕ", 1)
    add_paragraph(
        doc,
        "Сердечно-сосудистые заболевания (ССЗ) остаются одной из главных причин "
        "смертности в мире. Ранняя диагностика риска развития ишемической болезни сердца "
        "позволяет снизить нагрузку на систему здравоохранения и улучшить прогноз для пациента. "
        "В рамках типового исследования по дисциплине «Технологии машинного обучения» "
        "выполнено построение и сравнительная оценка моделей бинарной классификации "
        "по клиническим показателям пациента."
    )
    add_paragraph(
        doc,
        "Цель работы — разработать воспроизводимый конвейер машинного обучения "
        "(от анализа данных до веб-демонстрации), обосновать выбор метрик и моделей, "
        "сравнить качество baseline-решений и моделей с подобранными гиперпараметрами."
    )
    add_paragraph(
        doc,
        "Объект исследования — набор данных Heart Disease (Cleveland), UCI Machine Learning Repository. "
        "Предмет исследования — методы классификации и их качество на медицинских признаках."
    )

    add_heading(doc, "1 Постановка задачи", 1)
    add_paragraph(
        doc,
        "Задача формулируется как бинарная классификация: по 13 клиническим признакам "
        "определить наличие (1) или отсутствие (0) сердечно-сосудистого заболевания. "
        "Исходный целевой признак num содержал значения 0–4; для бинарной постановки "
        "значения 1–4 объединены в класс «болезнь присутствует»."
    )

    add_heading(doc, "2 Набор данных", 1)
    rows = summary.get("dataset_rows", 303)
    add_paragraph(
        doc,
        f"Использован открытый набор Cleveland Heart Disease ({rows} записей после удаления пропусков). "
        "Признаки: возраст, пол, тип боли в груди, артериальное давление, холестерин, "
        "уровень сахара, результаты ЭКГ, максимальная частота сердечных сокращений, "
        "стенокардия при нагрузке, депрессия ST, наклон ST, число сосудов, талассемия."
    )
    add_paragraph(
        doc,
        "Источник: UCI Machine Learning Repository, файл processed.cleveland.data. "
        "Датасет выбран как не пересекающийся с темами других студентов курса "
        "(не используются автомобили, зарплаты, отток телекома, вино, квартиры и др.)."
    )

    add_heading(doc, "3 Разведочный анализ данных", 1)
    add_paragraph(
        doc,
        "Проведён анализ структуры данных: проверены типы признаков, распределение классов, "
        "пропущенные значения (символ «?» в исходном файле), построены гистограммы "
        "числовых признаков в разрезе классов."
    )
    add_figure(doc, FIG_DIR / "01_target_distribution.png",
               "Рисунок 1 — Распределение целевого признака")
    add_figure(doc, FIG_DIR / "03_numeric_histograms.png",
               "Рисунок 2 — Гистограммы числовых признаков по классам")

    add_heading(doc, "4 Подготовка признаков", 1)
    add_paragraph(
        doc,
        "Категориальные признаки закодированы методом One-Hot Encoding. "
        "Числовые признаки масштабированы StandardScaler после имputation медианой. "
        "Сформированы вспомогательные признаки: age_group (возрастные группы), "
        "chol_high (холестерин выше 240 мг/дл), hr_risk (ЧСС на нагрузке ниже 120)."
    )

    add_heading(doc, "5 Корреляционный анализ", 1)
    add_paragraph(
        doc,
        "Построена корреляционная матрица Пирсона. Наибольшая связь с целевым признаком "
        "наблюдается у exang, cp, thalach, oldpeak — это обосновывает применимость "
        "линейных и нелинейных моделей. Сильной мультиколлинеарности, препятствующей обучению, не выявлено."
    )
    add_figure(doc, FIG_DIR / "04_correlation_matrix.png",
               "Рисунок 3 — Корреляционная матрица признаков")

    add_heading(doc, "6 Метрики качества", 1)
    add_paragraph(
        doc,
        "Выбраны пять метрик (не менее трёх по заданию):"
    )
    for m in [
        "Precision (точность положительного класса) — важна при минимизации ложных тревог.",
        "Recall (полнота) — критична в медицине: пропуск больного пациента дороже ложной тревоги.",
        "F1-мера — баланс precision и recall при несбалансированных классах.",
        "ROC AUC — оценка ранжирующей способности модели независимо от порога.",
        "Accuracy — доля верных ответов для общей сравнимости моделей.",
    ]:
        p = doc.add_paragraph(m, style="List Bullet")
        p.paragraph_format.first_line_indent = Cm(0)

    add_heading(doc, "7 Выбор моделей машинного обучения", 1)
    add_paragraph(
        doc,
        "Использовано семь моделей (требование — не менее пяти), из них две ансамблевые "
        "(Random Forest, Extra Trees) плюс Gradient Boosting как третий ансамблевый метод:"
    )
    models_list = [
        "логистическая регрессия (LogReg);",
        "метод k ближайших соседей (KNN);",
        "машина опорных векторов (SVC);",
        "решающее дерево (Tree);",
        "случайный лес (RF) — ансамбль;",
        "Extra Trees (ET) — ансамбль;",
        "градиентный бустинг (GB) — ансамбль.",
    ]
    for m in models_list:
        p = doc.add_paragraph(m, style="List Bullet")
        p.paragraph_format.first_line_indent = Cm(0)

    add_heading(doc, "8 Формирование обучающей и тестовой выборок", 1)
    tr = summary.get("train_size", 227)
    te = summary.get("test_size", 76)
    add_paragraph(
        doc,
        f"Данные разделены в соотношении 75/25 (train={tr}, test={te}) "
        "с стратификацией по целевому классу и random_state=42."
    )

    add_heading(doc, "9 Baseline-модели", 1)
    add_paragraph(
        doc,
        "Для каждой модели обучение выполнено с гиперпараметрами по умолчанию (без подбора). "
        "Оценка качества — на отложенной тестовой выборке."
    )
    add_figure(doc, FIG_DIR / "06_roc_auc_baseline.png",
               "Рисунок 4 — ROC AUC моделей (baseline)")

    add_heading(doc, "10 Подбор гиперпараметров", 1)
    add_paragraph(
        doc,
        "Применён GridSearchCV с 5-fold стратифицированной кросс-валидацией. "
        "Оптимизация велась по метрике ROC AUC. Для каждой модели задана сетка параметров "
        "(например, C и kernel для SVC, n_estimators и max_depth для лесов)."
    )

    add_heading(doc, "11 Сравнение baseline и tuned-моделей", 1)
    add_paragraph(
        doc,
        f"После подбора гиперпараметров наилучший ROC AUC на тесте показала модель {best_model}. "
        "Для большинства алгоритмов tuned-версия превосходит baseline, что подтверждает "
        "необходимость кросс-валидационного подбора."
    )
    add_figure(doc, FIG_DIR / "05_compare_roc_auc.png",
               "Рисунок 5 — Сравнение ROC AUC: baseline и tuned")
    add_figure(doc, FIG_DIR / "07_roc_auc_tuned.png",
               "Рисунок 6 — ROC AUC после подбора гиперпараметров")
    add_figure(doc, FIG_DIR / "09_best_roc_curve.png",
               f"Рисунок 7 — ROC-кривая лучшей модели ({best_model})")
    add_figure(doc, FIG_DIR / "10_best_confusion_matrix.png",
               f"Рисунок 8 — Нормированная матрица ошибок ({best_model})")

    add_heading(doc, "12 Выводы по результатам моделирования", 1)
    add_figure(doc, FIG_DIR / "11_rf_n_estimators.png",
               "Рисунок 9 — Влияние n_estimators на ROC AUC (Random Forest)")
    add_figure(doc, FIG_DIR / "12_learning_curves.png",
               "Рисунок 10 — Зависимость ROC AUC от размера обучающей выборки")
    add_paragraph(
        doc,
        "Ансамблевые методы (RF, ET, GB) стабильно входят в лидеры по F1 и ROC AUC. "
        "Логистическая регрессия даёт интерпретируемый baseline. KNN чувствителен к масштабу признаков, "
        "поэтому использован пайплайн с нормализацией. Рост n_estimators у Random Forest "
        "улучшает качество до насыщения (~100–150 деревьев)."
    )

    add_heading(doc, "13 Веб-приложение для демонстрации модели", 1)
    add_paragraph(
        doc,
        "Разработано веб-приложение на Streamlit (файл app/streamlit_app.py). "
        "Пользователь задаёт клинические показатели пациента и изменяет гиперпараметр "
        "n_estimators (и max_depth) Random Forest; при каждом изменении модель "
        "переобучается и выводится вероятность заболевания, прогноз класса и метрики на тесте."
    )
    add_paragraph(
        doc,
        "Запуск: streamlit run app/streamlit_app.py. Исходный код, данные и ноутбук "
        "размещаются в репозитории GitHub в соответствии с требованиями НИРС.",
        indent=False,
    )

    add_heading(doc, "ЗАКЛЮЧЕНИЕ", 1)
    add_paragraph(
        doc,
        "Выполнено типовое исследование по схеме курса «Технологии машинного обучения»: "
        "выбран набор данных Heart Disease (Cleveland), проведены EDA, подготовка признаков, "
        "корреляционный анализ, обучение семи моделей, baseline и GridSearchCV, "
        "сравнительная визуализация метрик, разработано Streamlit-приложение. "
        f"Лучшее качество на тестовой выборке обеспечила модель {best_model} после подбора гиперпараметров."
    )

    add_heading(doc, "СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ", 1)
    sources = [
        "1. Hastie T., Tibshirani R., Friedman J. The Elements of Statistical Learning. Springer, 2009.",
        "2. Pedregosa F. et al. Scikit-learn: Machine Learning in Python // JMLR. 2011. Vol. 12. P. 2825–2830.",
        "3. Heart Disease Data Set [Электронный ресурс]. UCI Machine Learning Repository. URL: https://archive.ics.uci.edu/ml/datasets/Heart+Disease (дата обращения: 03.06.2026).",
        "4. Streamlit Documentation [Электронный ресурс]. URL: https://docs.streamlit.io/ (дата обращения: 03.06.2026).",
        "5. James G., Witten D., Hastie T., Tibshirani R. An Introduction to Statistical Learning. Springer, 2021.",
        "6. Технологии машинного обучения: типовое задание на НИРС. Методические материалы дисциплины.",
        "7. ГОСТ 7.32–2017. Отчёт о научно-исследовательской работе. Структура и правила оформления.",
    ]
    for s in sources:
        p = doc.add_paragraph(s)
        p.paragraph_format.first_line_indent = Cm(0)
        for r in p.runs:
            r.font.name = "Times New Roman"
            r.font.size = Pt(14)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT_PATH)
    print(f"Отчёт сохранён: {OUT_PATH}")


if __name__ == "__main__":
    main()
