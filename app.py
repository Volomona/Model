
import sys
import io
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import streamlit as st
except ModuleNotFoundError:
    sys.exit("Error: Streamlit is not available. Please install and run locally: `streamlit run app.py`.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simulate_logistic(N0: float, r: float, K: float, T: int) -> np.ndarray:
    Ns = [N0]
    for _ in range(T):
        Ns.append(Ns[-1] + r * Ns[-1] * (1 - Ns[-1] / K))
    return np.array(Ns)

def simulate_ricker(N0: float, r: float, K: float, T: int) -> np.ndarray:
    Ns = [N0]
    for _ in range(T):
        Ns.append(Ns[-1] * np.exp(r * (1 - Ns[-1] / K)))
    return np.array(Ns)

def simulate_leslie(N0_vec: list, fertility: list, survival: list, T: int) -> np.ndarray:
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
    Ns = [N0] * (tau + 1)
    for t in range(tau, T + tau):
        Ns.append(Ns[t] * np.exp(r * (1 - Ns[t - tau] / K)))
    return np.array(Ns)

def simulate_stochastic(base_sim, *args, sigma: float = 0.1, repeats: int = 100) -> np.ndarray:
    runs = []
    progress = st.progress(0)
    for i in range(repeats):
        traj = base_sim(*args)
        noise = np.random.normal(0, sigma, size=traj.shape)
        runs.append(np.clip(traj + noise, 0, None))
        progress.progress((i + 1) / repeats)
    return np.array(runs)

def export_csv(data, filename):
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Скачать данные CSV",
        data=csv,
        file_name=f"{filename}.csv",
        mime="text/csv"
    )

st.set_page_config(page_title="Population Dynamics Simulator", layout="wide")
st.title("🌱 Симулятор Популяционной Динамики")

model_info = {
    "Логистический рост": "Классическая логистическая карта с предельной численностью K.",
    "Модель Рикера": "Экспоненциальный рост с зависимостью от плотности (Рикер).",
    "Модель Лесли": "Возрастная структура модели через матрицу Лесли.",
    "Модель с задержкой": "Популяция зависит от прошлого состояния (задержка τ).",
    "Стохастическая симуляция": "Добавляет гауссов шум к нескольким запускам.",
}
st.sidebar.info("Выберите модель и установите параметры ниже.")
model = st.sidebar.selectbox("Выберите модель:", list(model_info.keys()))
st.sidebar.caption(model_info[model])
T = st.sidebar.number_input("Шаги времени (T)", min_value=1, max_value=500, value=100)

if model in ["Логистический рост", "Модель Рикера", "Модель с задержкой", "Стохастическая симуляция"]:
    num_configs = st.sidebar.slider("Сколько конфигураций сравнить?", 1, 5, 2)
    param_sets = []
    for i in range(num_configs):
        with st.sidebar.expander(f"Конфигурация {i+1}"):
            N0 = st.number_input(f"N0 ({i+1})", min_value=0.0, value=10.0, key=f"N0_{i}")
            r = st.number_input(f"r ({i+1})", min_value=0.0, value=0.1, key=f"r_{i}")
            K = st.number_input(f"K ({i+1})", min_value=1.0, value=100.0, key=f"K_{i}")
            if model == "Модель с задержкой":
                tau = st.number_input(f"τ ({i+1})", min_value=1, max_value=10, value=1, key=f"tau_{i}")
                param_sets.append((N0, r, K, tau))
            else:
                param_sets.append((N0, r, K))

if model == "Модель Лесли":
    n = st.sidebar.number_input("Число возрастных классов", min_value=2, max_value=10, value=3)
    with st.sidebar.expander("Коэффициенты рождаемости (f_i)"):
        fertility = [st.number_input(f"f_{i}", min_value=0.0, value=0.5) for i in range(n)]
    with st.sidebar.expander("Вероятности выживания (s_i)"):
        survival = [st.number_input(f"s_{i}", min_value=0.0, max_value=1.0, value=0.8) for i in range(n-1)]
    with st.sidebar.expander("Начальная популяция по возрастным классам"):
        N0_vec = [st.number_input(f"N0_{i}", min_value=0.0, value=10.0) for i in range(n)]

if model == "Стохастическая симуляция":
    repeats = st.sidebar.number_input("Число повторений", min_value=1, max_value=200, value=100)
    sigma = st.sidebar.slider("Шум (σ)", min_value=0.0, max_value=1.0, value=0.1)
    base_model = st.sidebar.selectbox("Основная модель:", ["Logistic", "Ricker"])

if st.sidebar.button("Симулировать"):
    with st.spinner("Симуляция..."):
        if model == "Логистический рост":
            results = []
            labels = []
            for idx, (N0, r, K) in enumerate(param_sets):
                traj = simulate_logistic(N0, r, K, T)
                results.append(traj)
                labels.append(f"Конфигурация {idx+1}")
            df = pd.DataFrame(np.array(results).T, columns=labels)
            st.subheader("Логистический рост — Сравнение")
            st.line_chart(df)
            export_csv(df, "logistic_comparison")

        elif model == "Модель Рикера":
            results = []
            labels = []
            for idx, (N0, r, K) in enumerate(param_sets):
                traj = simulate_ricker(N0, r, K, T)
                results.append(traj)
                labels.append(f"Конфигурация {idx+1}")
            df = pd.DataFrame(np.array(results).T, columns=labels)
            st.subheader("Модель Рикера — Сравнение")
            st.line_chart(df)
            export_csv(df, "ricker_comparison")

        elif model == "Модель с задержкой":
            results = []
            labels = []
            for idx, (N0, r, K, tau) in enumerate(param_sets):
                traj = simulate_delay(N0, r, K, T, tau)
                results.append(traj)
                labels.append(f"Конфигурация {idx+1}")
            df = pd.DataFrame(np.array(results).T, columns=labels)
            st.subheader("Модель с задержкой — Сравнение")
            st.line_chart(df)
            export_csv(df, "delay_comparison")

        elif model == "Стохастическая симуляция":
            base_sim = simulate_ricker if base_model == "Ricker" else simulate_logistic
            results = []
            labels = []
            for idx, (N0, r, K) in enumerate(param_sets):
                sim = simulate_stochastic(base_sim, N0, r, K, T, sigma=sigma, repeats=repeats)
                mean_traj = sim.mean(axis=0)
                results.append(mean_traj)
                labels.append(f"Конфигурация {idx+1}")
            df = pd.DataFrame(np.array(results).T, columns=labels)
            st.subheader("Стохастическая модель — Средние траектории")
            st.line_chart(df)
            export_csv(df, "stochastic_comparison")

        elif model == "Модель Лесли":
            history = simulate_leslie(N0_vec, fertility, survival, T)
            df = pd.DataFrame(history, columns=[f"Возраст {i}" for i in range(n)])
            st.subheader("Модель Лесли")
            st.line_chart(df)
            L = np.zeros((n, n))
            L[0, :] = fertility
            for i in range(1, n):
                L[i, i-1] = survival[i-1]
            lambda_val = np.max(np.real(np.linalg.eigvals(L)))
            export_csv(history, "leslie_matrix")
            st.write(f"Доминирующее собственное значение λ = {lambda_val:.3f}")


# Footer
st.sidebar.markdown("---")
st.sidebar.info("Разработано Лией Ахметовой — v1.0")
