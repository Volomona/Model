import sys
import io
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame(data)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Скачать данные CSV",
        data=csv,
        file_name=f"{filename}.csv",
        mime="text/csv"
    )

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

st.sidebar.markdown("### Число временных шагов")
T = st.sidebar.number_input("Шаги времени (T)", min_value=1, max_value=500, value=100)

if model == "Модель Лесли":
    n = st.sidebar.number_input("Число возрастных классов", min_value=2, max_value=10, value=3)
    with st.sidebar.expander("Коэффициенты рождаемости (f_i)"):
        fertility = [st.number_input(f"f_{i}", min_value=0.0, value=0.5, key=f"fertility_{i}") for i in range(n)]
    with st.sidebar.expander("Вероятности выживания (s_i)"):
        survival = [st.number_input(f"s_{i}", min_value=0.0, max_value=1.0, value=0.8, key=f"survival_{i}") for i in range(n-1)]
    with st.sidebar.expander("Начальная популяция по возрастным классам"):
        N0_vec = [st.number_input(f"N0_{i}", min_value=0.0, value=10.0, key=f"N0_vec_{i}") for i in range(n)]

elif model == "Модель с задержкой":
    st.sidebar.markdown("### Конфигурации параметров")
    tau = st.sidebar.slider("Задержка (τ)", min_value=1, max_value=10, value=1)
    num_configs = st.sidebar.number_input("Число конфигураций", min_value=1, max_value=5, value=2)
    configs = []
    for i in range(num_configs):
        with st.sidebar.expander(f"Конфигурация #{i+1}"):
            N0 = st.number_input(f"N0 [{i+1}]", value=10.0, key=f"N0_delay_{i}")
            r = st.number_input(f"r [{i+1}]", value=0.1, key=f"r_delay_{i}")
            K = st.number_input(f"K [{i+1}]", value=100.0, key=f"K_delay_{i}")
            configs.append({'N0': N0, 'r': r, 'K': K})

elif model == "Стохастическая симуляция":
    repeats = st.sidebar.number_input("Число повторений", min_value=1, max_value=200, value=100)
    base_model = st.sidebar.selectbox("Основная модель:", ["Logistic", "Ricker"])
    st.sidebar.markdown("### Конфигурации стохастической модели")
    num_configs = st.sidebar.number_input("Число конфигураций", min_value=1, max_value=5, value=2)
    configs = []
    for i in range(num_configs):
        with st.sidebar.expander(f"Конфигурация #{i+1}"):
            N0 = st.number_input(f"N0 [{i+1}]", value=10.0, key=f"N0_stoch_{i}")
            r = st.number_input(f"r [{i+1}]", value=0.1, key=f"r_stoch_{i}")
            K = st.number_input(f"K [{i+1}]", value=100.0, key=f"K_stoch_{i}")
            sigma = st.slider(f"σ (шум) [{i+1}]", min_value=0.0, max_value=1.0, value=0.1, key=f"sigma_stoch_{i}")
            configs.append({'N0': N0, 'r': r, 'K': K, 'sigma': sigma})

else:
    st.sidebar.markdown("### Конфигурации параметров")
    num_configs = st.sidebar.number_input("Число конфигураций", min_value=1, max_value=5, value=2)
    configs = []
    for i in range(num_configs):
        with st.sidebar.expander(f"Конфигурация #{i+1}"):
            N0 = st.number_input(f"N0 [{i+1}]", value=10.0, key=f"N0_{i}")
            r = st.number_input(f"r [{i+1}]", value=0.1, key=f"r_{i}")
            K = st.number_input(f"K [{i+1}]", value=100.0, key=f"K_{i}")
            configs.append({'N0': N0, 'r': r, 'K': K})

if st.sidebar.button("Симулировать"):
    with st.spinner("Симуляция..."):
        if model == "Логистический рост":
            st.subheader("Сравнение конфигураций логистического роста")
            fig, ax = plt.subplots()
            for idx, conf in enumerate(configs):
                traj = simulate_logistic(conf['N0'], conf['r'], conf['K'], T)
                ax.plot(traj, label=f"Конфигурация #{idx+1}")
            ax.set_title("Логистический рост")
            ax.set_xlabel("Время")
            ax.set_ylabel("Популяция")
            ax.legend()
            st.pyplot(fig)

        elif model == "Модель Рикера":
            st.subheader("Сравнение конфигураций модели Рикера")
            fig, ax = plt.subplots()
            for idx, conf in enumerate(configs):
                traj = simulate_ricker(conf['N0'], conf['r'], conf['K'], T)
                ax.plot(traj, label=f"Конфигурация #{idx+1}")
            ax.set_title("Модель Рикера")
            ax.set_xlabel("Время")
            ax.set_ylabel("Популяция")
            ax.legend()
            st.pyplot(fig)

        elif model == "Модель с задержкой":
            st.subheader("Сравнение конфигураций модели с задержкой")
            fig, ax = plt.subplots()
            for idx, conf in enumerate(configs):
                traj = simulate_delay(conf['N0'], conf['r'], conf['K'], T, tau)
                ax.plot(traj, label=f"Конфигурация #{idx+1}")
            ax.set_title(f"Модель с задержкой τ={tau}")
            ax.set_xlabel("Время")
            ax.set_ylabel("Популяция")
            ax.legend()
            st.pyplot(fig)

        elif model == "Модель Лесли":
            history = simulate_leslie(N0_vec, fertility, survival, T)
            df = pd.DataFrame(history, columns=[f"Возраст {i}" for i in range(n)])
            st.subheader("Модель Лесли")
            st.line_chart(df)
            L = np.zeros((n, n)); L[0, :] = fertility
            for i in range(1, n): L[i, i-1] = survival[i-1]
            lambda_val = np.max(np.real(np.linalg.eigvals(L)))
            export_csv(history, 'leslie_matrix_data')
            st.write(f"Доминирующее собственное значение λ = {lambda_val:.3f}")

        elif model == "Стохастическая симуляция":
            base_sim = simulate_ricker if base_model == 'Ricker' else simulate_logistic
            st.subheader("Сравнение стохастических конфигураций")
            fig, ax = plt.subplots()
            for idx, conf in enumerate(configs):
                results = simulate_stochastic(base_sim, conf['N0'], conf['r'], conf['K'], T,
                                              sigma=conf['sigma'], repeats=repeats)
                mean_traj = results.mean(axis=0)
                ax.plot(mean_traj, label=f"Конфигурация #{idx+1} (σ={conf['sigma']:.2f})")
            ax.set_title("Средние траектории стохастической симуляции")
            ax.set_xlabel("Время")
            ax.set_ylabel("Популяция")
            ax.legend()
            st.pyplot(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Разработано Лией Ахметовой — v1.0")
