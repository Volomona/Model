import sys
import logging

import numpy as np
import pandas as pd

# Streamlit import with exit if unavailable
try:
    import streamlit as st
except ModuleNotFoundError:
    sys.exit("Error: Streamlit is not available. Please install and run locally: `streamlit run app.py`.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==== Simulation functions ==== #
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

def simulate_stochastic(base_sim, N0: float, r: float, K: float, T: int,
                        sigma: float = 0.1, repeats: int = 100) -> np.ndarray:
    runs = []
    progress = st.progress(0)
    for i in range(repeats):
        traj = base_sim(N0, r, K, T)
        noise = np.random.normal(0, sigma, size=traj.shape)
        runs.append(np.clip(traj + noise, 0, None))
        progress.progress((i + 1) / repeats)
    return np.array(runs)

# ==== Streamlit UI ==== #
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

st.sidebar.markdown("### Общие параметры")
T = st.sidebar.number_input("Шаги времени (T)", min_value=1, max_value=500, value=100)

# Выбор количества конфигураций
num_configs = st.sidebar.number_input("Число конфигураций для сравнения", min_value=1, max_value=10, value=1)

configs = []

if model != "Модель Л Leslie":
    for i in range(num_configs):
        st.sidebar.markdown(f"---\n### Конфигурация {i+1}")
        N0 = st.sidebar.number_input(f"Начальная популяция N0 (конфигурация {i+1})", min_value=0.0, value=10.0, key=f"N0_{i}")
        r = st.sidebar.number_input(f"Темп роста r (конфигурация {i+1})", min_value=0.0, value=0.1, key=f"r_{i}")
        K = st.sidebar.number_input(f"Емкость K (конфигурация {i+1})", min_value=1.0, value=100.0, key=f"K_{i}")

        config = {'N0': N0, 'r': r, 'K': K}

        if model == "Модель с задержкой":
            tau = st.sidebar.slider(f"Задержка (τ) (конфигурация {i+1})", min_value=1, max_value=10, value=1, key=f"tau_{i}")
            config['tau'] = tau

        if model == "Стохастическая симуляция":
            sigma = st.sidebar.slider(f"Шум (σ) (конфигурация {i+1})", min_value=0.0, max_value=1.0, value=0.1, key=f"sigma_{i}")
            repeats = st.sidebar.number_input(f"Число повторений (конфигурация {i+1})", min_value=1, max_value=200, value=100, key=f"repeats_{i}")
            base_model = st.sidebar.selectbox(f"Основная модель (конфигурация {i+1})", ["Logistic", "Ricker"], key=f"base_model_{i}")
            config['sigma'] = sigma
            config['repeats'] = repeats
            config['base_model'] = base_model

        configs.append(config)

elif model == "Модель Лесли":
    n = st.sidebar.number_input("Число возрастных классов", min_value=2, max_value=10, value=3)
    with st.sidebar.expander("Коэффициенты рождаемости (f_i)"):
        fertility = [st.number_input(f"f_{i}", min_value=0.0, value=0.5) for i in range(n)]
    with st.sidebar.expander("Вероятности выживания (s_i)"):
        survival = [st.number_input(f"s_{i}", min_value=0.0, max_value=1.0, value=0.8) for i in range(n-1)]
    with st.sidebar.expander("Начальная популяция по возрастным классам"):
        N0_vec = [st.number_input(f"N0_{i}", min_value=0.0, value=10.0) for i in range(n)]

if st.sidebar.button("Симулировать"):
    with st.spinner("Симуляция..."):
        if model == "Логистический рост":
            st.subheader("Логистический рост — сравнение конфигураций")
            all_trajs = {}
            for i, conf in enumerate(configs):
                traj = simulate_logistic(conf['N0'], conf['r'], conf['K'], T)
                all_trajs[f"Конфигурация {i+1}"] = traj
            df = pd.DataFrame(all_trajs)
            st.line_chart(df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Скачать CSV для всех конфигураций", data=csv,
                               file_name="logistic_growth_multiple_configs.csv", mime="text/csv")

        elif model == "Модель Рикера":
            st.subheader("Модель Рикера — сравнение конфигураций")
            all_trajs = {}
            for i, conf in enumerate(configs):
                traj = simulate_ricker(conf['N0'], conf['r'], conf['K'], T)
                all_trajs[f"Конфигурация {i+1}"] = traj
            df = pd.DataFrame(all_trajs)
            st.line_chart(df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Скачать CSV для всех конфигураций", data=csv,
                               file_name="ricker_model_multiple_configs.csv", mime="text/csv")

        elif model == "Модель с задержкой":
            st.subheader("Модель с задержкой — сравнение конфигураций")
            all_trajs = {}
            for i, conf in enumerate(configs):
                traj = simulate_delay(conf['N0'], conf['r'], conf['K'], T, conf['tau'])
                all_trajs[f"Конфигурация {i+1}"] = traj
            df = pd.DataFrame(all_trajs)
            st.line_chart(df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Скачать CSV для всех конфигураций", data=csv,
                               file_name="delay_model_multiple_configs.csv", mime="text/csv")

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
            st.write(f"Доминирующее собственное значение λ = {lambda_val:.3f}")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Скачать CSV", data=csv, file_name="leslie_model.csv", mime="text/csv")

        elif model == "Стохастическая симуляция":
            st.subheader("Стохастическая симуляция — сравнение конфигураций")
            all_means = {}
            for i, conf in enumerate(configs):
                base_func = simulate_logistic if conf['base_model'] == "Logistic" else simulate_ricker
                runs = simulate_stochastic(base_func, conf['N0'], conf['r'], conf['K'], T,
                                          sigma=conf['sigma'], repeats=conf['repeats'])
                mean_traj = np.mean(runs, axis=0)
                all_means[f"Конфигурация {i+1}"] = mean_traj
            df = pd.DataFrame(all_means)
            st.line_chart(df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Скачать CSV для всех конфигураций", data=csv,
                               file_name="stochastic_model_multiple_configs.csv", mime="text/csv")

else:
    st.info("Настройте параметры и нажмите кнопку 'Симулировать' в боковой панели.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Разработано Лией Ахметовой — v1.0")
