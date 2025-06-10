import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import g4f
from scipy.optimize import minimize
import pdfkit

# -------------------------------
# Настройка логирования
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Simulation functions
# -------------------------------
def simulate_logistic(N0: float, r: float, K: float, T: int) -> np.ndarray:
    """Логистическая карта"""
    Ns = [N0]
    for _ in range(T):
        Ns.append(Ns[-1] + r * Ns[-1] * (1 - Ns[-1] / K))
    return np.array(Ns)

def simulate_ricker(N0: float, r: float, K: float, T: int) -> np.ndarray:
    """Модель Рикера"""
    Ns = [N0]
    for _ in range(T):
        Ns.append(Ns[-1] * np.exp(r * (1 - Ns[-1] / K)))
    return np.array(Ns)

def simulate_leslie(N0_vec: list, fertility: list, survival: list, T: int) -> np.ndarray:
    """Возрастная модель Лесли"""
    n = len(N0_vec)
    N = np.array(N0_vec, dtype=float)
    history = [N.copy()]
    L = np.zeros((n, n))
    L[0, :] = fertility
    for i in range(1, n):
        L[i, i-1] = survival[i-1]
    for _ in range(T):
        N = L.dot(N)
        history.append(N.copy())
    return np.array(history)

def simulate_delay(N0: float, r: float, K: float, T: int, tau: int) -> np.ndarray:
    """Модель с задержкой"""
    Ns = [N0] * (tau + 1)
    for t in range(tau, T + tau):
        N_next = Ns[t] * np.exp(r * (1 - Ns[t - tau] / K))
        Ns.append(N_next)
    return np.array(Ns[:T + 1])

def simulate_stochastic(base_sim, *args, sigma: float = 0.1, repeats: int = 100) -> np.ndarray:
    """Стохастическая симуляция"""
    runs = []
    progress = st.progress(0)
    for i in range(repeats):
        traj = base_sim(*args)
        noise = np.random.normal(0, sigma, size=traj.shape)
        runs.append(np.clip(traj + noise, 0, None))
        progress.progress((i + 1) / repeats)
    return np.array(runs)

# -------------------------------
# Export and Analysis
# -------------------------------
def export_csv(data: np.ndarray, filename: str, model_type: str, params_str: str):
    """Сохраняет данные в CSV и предлагает скачать"""
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Скачать данные CSV",
        data=csv,
        file_name=f"{filename}.csv",
        mime="text/csv",
        key='download_csv'
    )

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Population Dynamics Simulator", layout="wide")
st.title("🌱 Симулятор Популяционной Динамики")

# Инициализация состояния
if 'ts' not in st.session_state:
    st.session_state.ts = None
if 'model' not in st.session_state:
    st.session_state.model = None

# Sidebar: выбор модели и параметров
models = {
    "Логистический рост": simulate_logistic,
    "Модель Рикера": simulate_ricker,
    "Модель Лесли": simulate_leslie,
    "Модель с задержкой": simulate_delay,
    "Стохастическая симуляция": simulate_stochastic,
}
model_name = st.sidebar.selectbox("Выберите модель:", list(models.keys()))
T = st.sidebar.slider("Шаги времени (T)", 10, 500, 100)
N0 = st.sidebar.number_input("Начальная популяция N0", 0.0, 1000.0, 10.0)
r = st.sidebar.number_input("Темп роста r", 0.0, 5.0, 0.5)
K = st.sidebar.number_input("Емкость среды K", 1.0, 1000.0, 100.0)

# Специфичные параметры
if model_name == "Модель с задержкой":
    tau = st.sidebar.number_input("Задержка τ", 1, 10, 1)
elif model_name == "Стохастическая симуляция":
    sigma = st.sidebar.number_input("Интенсивность шума σ", 0.0, 1.0, 0.1)
    repeats = st.sidebar.number_input("Повторения", 1, 200, 50)
elif model_name == "Модель Лесли":
    n = st.sidebar.number_input("Классов (n)", 2, 10, 3)
    fertility = [st.sidebar.number_input(f"f_{i}", 0.0, 1.0, 0.5) for i in range(n)]
    survival = [st.sidebar.number_input(f"s_{i}", 0.0, 1.0, 0.8) for i in range(n-1)]
    N0_vec = [st.sidebar.number_input(f"N0_{i}", 0.0, 1000.0, 10.0) for i in range(n)]

# Кнопка симуляции
if st.sidebar.button("Симулировать"):
    if model_name == "Модель с задержкой":
        ts = models[model_name](N0, r, K, T, tau)
    elif model_name == "Стохастическая симуляция":
        ts = models[model_name](simulate_logistic, N0, r, K, T, sigma=sigma, repeats=repeats)
    elif model_name == "Модель Лесли":
        ts = models[model_name](N0_vec, fertility, survival, T)
    else:
        ts = models[model_name](N0, r, K, T)
    st.session_state.ts = ts
    st.session_state.model = model_name

# Отображение результатов
if st.session_state.ts is not None:
    st.subheader(f"Результаты: {st.session_state.model}")
    df_plot = pd.DataFrame(
        st.session_state.ts if st.session_state.ts.ndim == 1 else st.session_state.ts
    )
    st.line_chart(df_plot)

    # Кнопка скачивания CSV
    export_csv(
        st.session_state.ts,
        st.session_state.model.replace(" ", "_"),
        st.session_state.model,
        ""
    )

    # Кнопка анализа GPT-4
    if st.button("Запросить анализ GPT-4"):
        snippet = str(st.session_state.ts.flatten()[:10]) + "..."
        response = g4f.ChatCompletion.create(
            model=g4f.models.gpt_4,
            messages=[{
                "role": "user",
                "content": (
                    f"Вы - научный сотрудник. Проанализируйте результаты симуляции.\n"
                    f"Тип модели: {st.session_state.model}\n"
                    f"Данные (первые 10 точек): {snippet}"
                )
            }]
        )
        st.subheader("Анализ GPT-4:")
        st.write(response)

st.sidebar.markdown("---")
st.sidebar.info("Разработано Лией Ахметовой")
