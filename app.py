import os
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# ---------------- НАСТРОЙКА СТРАНИЦЫ ----------------

st.set_page_config(
    page_title="Прогнозирование трафика, температуры и продаж",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Прогнозирование трафика, температуры и продаж")
st.caption("Готовый учебный сайт-проект на Python + Streamlit + scikit-learn")

st.markdown(
    """
Этот сайт показывает полный цикл **прогнозирования временных рядов**:

1. Загрузка или использование **готового примера данных**
2. Генерация признаков (день недели, месяц, лаги)
3. Обучение модели `RandomForestRegressor`
4. Оценка качества (MAE, RMSE)
5. Построение прогноза на будущее и визуализация

Проект можно использовать как работу по теме  
**«Прогнозирование трафика / температуры / продаж с помощью машинного обучения»**.
"""
)

# ---------------- ФУНКЦИИ ----------------


@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


@st.cache_data
def load_example_data() -> pd.DataFrame:
    """
    Загружаем пример данных из sample_data.csv, если он есть.
    Иначе генерируем синтетические данные "продаж".
    """
    path = "sample_data.csv"
    if os.path.exists(path):
        return pd.read_csv(path)

    dates = pd.date_range("2024-01-01", periods=180)
    rng = np.random.default_rng(42)

    # базовый тренд + сезонность + шум
    trend = np.linspace(100, 140, len(dates))
    season = 10 * np.sin(np.linspace(0, 6 * np.pi, len(dates)))
    noise = rng.normal(0, 5, len(dates))

    sales = trend + season + noise
    return pd.DataFrame({"date": dates, "value": np.round(sales, 2)})


def build_feature_table(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    add_lags: bool = True,
) -> pd.DataFrame:
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col)

    # календарные признаки
    data["dayofweek"] = data[date_col].dt.dayofweek
    data["month"] = data[date_col].dt.month
    data["day"] = data[date_col].dt.day

    # лаги целевой переменной
    if add_lags:
        data["lag_1"] = data[target_col].shift(1)
        data["lag_7"] = data[target_col].shift(7)

    data = data.dropna()
    return data


def split_train_test(
    data: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    feature_cols = [c for c in data.columns if c not in [target_col]]
    # исключаем саму дату, если она есть в признаках
    feature_cols = [c for c in feature_cols if "date" not in c.lower()]

    X = data[feature_cols]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return mae, rmse, y_pred


def make_forecast(
    model: RandomForestRegressor,
    history: pd.DataFrame,
    date_col: str,
    target_col: str,
    horizon: int,
) -> pd.DataFrame:
    df = history.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    last_date = df[date_col].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon)

    rows = []
    for dt in future_dates:
        rows.append(
            {
                date_col: dt,
                "dayofweek": dt.dayofweek,
                "month": dt.month,
                "day": dt.day,
            }
        )
    future = pd.DataFrame(rows)

    full = pd.concat([df[[date_col, target_col]], future[[date_col]]], ignore_index=True)
    full = full.sort_values(date_col).reset_index(drop=True)

    # протягиваем последние значения
    full[target_col] = full[target_col].ffill()

    full["lag_1"] = full[target_col].shift(1)
    full["lag_7"] = full[target_col].shift(7)

    future = future.merge(full[[date_col, "lag_1", "lag_7"]], on=date_col, how="left")

    future["lag_1"] = future["lag_1"].ffill()
    future["lag_7"] = future["lag_7"].ffill()

    feature_cols = [c for c in future.columns if c not in [date_col, target_col]]
    X_future = future[feature_cols]

    future[target_col + "_pred"] = model.predict(X_future)

    return future[[date_col, target_col + "_pred"]]


# ---------------- UI: БОКОВАЯ ПАНЕЛЬ ----------------

with st.sidebar:
    st.header("⚙️ Настройки")
    scenario = st.selectbox(
        "Тип сценария",
        [
            "Прогноз продаж",
            "Прогноз трафика (посещения / запросы)",
            "Прогноз температуры",
        ],
    )
    horizon = st.slider("Горизонт прогноза (дней)", 7, 60, 21)
    st.markdown(
        """
        **Подсказка:**
        1. Сначала попробуйте пример данных.
        2. Потом загрузите свой CSV.
        """
    )

# ---------------- БЛОК ДАННЫХ ----------------

st.subheader("1. Данные")

tab_example, tab_upload = st.tabs(["📘 Пример данных", "📂 Загрузить свой CSV"])

with tab_example:
    example_df = load_example_data()
    st.write("Пример данных (можно сразу использовать для обучения):")
    st.dataframe(example_df.head())

    csv_bytes = example_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Скачать пример данных (sample_data.csv)",
        data=csv_bytes,
        file_name="sample_data.csv",
        mime="text/csv",
    )

with tab_upload:
    uploaded = st.file_uploader(
        "Загрузите CSV (обязательно: колонка с датой и числовая колонка)",
        type=["csv"],
    )

use_example_only = uploaded is None
if use_example_only:
    df_raw = example_df.copy()
    st.info("Используется пример данных.")
else:
    df_raw = load_csv(uploaded)
    st.success("Используется загруженный CSV-файл.")

st.write("Первые строки данных:")
st.dataframe(df_raw.head())

if df_raw.shape[1] < 2:
    st.error("Нужно минимум 2 колонки: дата и числовой показатель.")
    st.stop()

# ---------------- ПОДГОТОВКА ДАННЫХ ----------------

st.subheader("2. Подготовка данных")

date_col = st.selectbox("Колонка с датой", options=df_raw.columns, index=0)

# разумный выбор целевой по названию
target_default = 1
for i, col in enumerate(df_raw.columns):
    if col.lower() in ["value", "sales", "traffic", "temperature", "target"]:
        target_default = i
        break

target_col = st.selectbox(
    "Целевая колонка (что прогнозируем)", options=df_raw.columns, index=target_default
)

if date_col == target_col:
    st.error("Колонка с датой и целевая колонка должны быть разными.")
    st.stop()

try:
    data_features = build_feature_table(df_raw, date_col, target_col)
except Exception as e:
    st.error(f"Ошибка при обработке данных: {e}")
    st.stop()

if len(data_features) < 30:
    st.warning("Мало данных (< 30 строк) — качество прогноза может быть нестабильным.")

st.write("Признаки для модели (часть):")
st.dataframe(data_features.head())

# ---------------- МОДЕЛЬ ----------------

st.subheader("3. Обучение модели и качество")

X_train, X_test, y_train, y_test = split_train_test(data_features, target_col)
model = train_model(X_train, y_train)
mae, rmse, y_pred_test = evaluate_model(model, X_test, y_test)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("MAE", f"{mae:,.3f}")
with c2:
    st.metric("RMSE", f"{rmse:,.3f}")
with c3:
    st.metric("Размер тестовой выборки", len(X_test))

# ---------------- ВИЗУАЛИЗАЦИЯ ----------------

st.subheader("4. История и прогноз")

history_chart = df_raw[[date_col, target_col]].copy()
history_chart[date_col] = pd.to_datetime(history_chart[date_col])
history_chart = history_chart.sort_values(date_col)

st.markdown("**Исторические данные:**")
st.line_chart(history_chart.set_index(date_col)[target_col], height=250)

forecast_df = make_forecast(
    model=model,
    history=df_raw[[date_col, target_col]],
    date_col=date_col,
    target_col=target_col,
    horizon=horizon,
)

st.markdown(f"**Прогноз на следующие {horizon} дней:**")
st.dataframe(forecast_df)

st.line_chart(
    forecast_df.set_index(date_col)[f"{target_col}_pred"],
    height=250,
)

# ---------------- ОПИСАНИЕ ДЛЯ ОТЧЕТА ----------------

st.markdown(
    """
---

### 🧾 Как можно описать проект на защите

- **Цель работы:** разработка веб-приложения для прогнозирования временных рядов  
  (продажи, трафик, температура) с помощью методов машинного обучения.
- **Инструменты:** Python, библиотеки `pandas`, `numpy`, `scikit-learn`, фреймворк `Streamlit`.
- **Модель:** ансамблевая модель `RandomForestRegressor` для регрессии.
- **Входные данные:** таблица (CSV) с двумя основными колонками:
  - дата (`date`)
  - числовой показатель (продажи / трафик / температура).
- **Результат:** пользователь загружает данные, система:
  1. строит признаки (календарные + лаги),
  2. обучает модель на истории,
  3. показывает качество (MAE, RMSE),
  4. строит прогноз на выбранный горизонт в днях.

Этот проект можно использовать как готовый пример практического применения ML
к задачам прогнозирования в бизнесе и аналитике.
"""
)
