import sys
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

# Streamlit import with exit if unavailable
try:
    import streamlit as st
except ModuleNotFoundError:
    sys.exit("Error: Streamlit is not available. Please install and run locally: `streamlit run app.py`.")

# ✅ set_page_config must come first
st.set_page_config(page_title="Population Dynamics Simulator", layout="wide")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==== Language selection ====
lang = st.sidebar.selectbox("🌐 Language / Язык", ["🇬🇧 English", "🇷🇺 Русский"])
locale = "en" if "English" in lang else "ru"

translations = {
    "en": {
        "title": "🌱 Population Dynamics Simulator",
        "select_model": "Choose model:",
        "common_params": "Common Parameters",
        "simulate": "Simulate",
        "time_steps": "Time steps (T)",
        "initial_N0": "Initial population N0",
        "growth_r": "Growth rate r",
        "carrying_K": "Carrying capacity K",
        "download_plot": "Download plot PNG",
        "download_csv": "Download data CSV",
        "mean_trajectory": "Mean trajectory:",
        "simulate_btn": "Simulate",
        "footer": "Developed by Liya Akhmetova — v1.0"
    },
    "ru": {
        "title": "🌱 Симулятор динамики популяции",
        "select_model": "Выберите модель:",
        "common_params": "Общие параметры",
        "simulate": "Симуляция",
        "time_steps": "Шаги моделирования (T)",
        "initial_N0": "Начальная популяция N₀",
        "growth_r": "Темп роста r",
        "carrying_K": "Емкость среды K",
        "download_plot": "Скачать график (PNG)",
        "download_csv": "Скачать данные (CSV)",
        "mean_trajectory": "Средняя траектория:",
        "simulate_btn": "Симулировать",
        "footer": "Разработано: Лия Ахметова — v1.0"
    }
}
_ = translations[locale]

# ==== Simulation functions ====
def simulate_logistic(N0, r, K, T):
    Ns = [N0]
    for _ in range(T):
        Ns.append(Ns[-1] + r * Ns[-1] * (1 - Ns[-1] / K))
    return np.array(Ns)

def simulate_ricker(N0, r, K, T):
    Ns = [N0]
    for _ in range(T):
        Ns.append(Ns[-1] * np.exp(r * (1 - Ns[-1] / K)))
    return np.array(Ns)

def simulate_leslie(N0_vec, fertility, survival, T):
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

def simulate_delay(N0, r, K, T, tau):
    Ns = [N0] * (tau + 1)
    for t in range(tau, T + tau):
        Ns.append(Ns[t] * np.exp(r * (1 - Ns[t - tau] / K)))
    return np.array(Ns)

def simulate_stochastic(base_sim, *args, sigma=0.1, repeats=100):
    runs = []
    for _ in range(repeats):
        traj = base_sim(*args)
        noise = np.random.normal(0, sigma, size=traj.shape)
        runs.append(traj + noise)
    return np.array(runs)

# ==== UI ====
st.title(_["title"])

model = st.sidebar.selectbox(_["select_model"], [
    "Logistic Growth", "Ricker Model", "Leslie Matrix", "Delay Model", "Stochastic"
])

st.sidebar.markdown(f"### {_['common_params']}")
T = st.sidebar.number_input(_["time_steps"], min_value=1, max_value=1000, value=100)

common = {}
if model != "Leslie Matrix":
    common['N0'] = st.sidebar.number_input(_["initial_N0"], value=10.0)
    common['r'] = st.sidebar.number_input(_["growth_r"], value=0.1)
    common['K'] = st.sidebar.number_input(_["carrying_K"], value=100.0)

if model == "Delay Model":
    tau = st.sidebar.slider("Delay (τ)", min_value=1, max_value=10, value=1)
elif model == "Leslie Matrix":
    n = st.sidebar.number_input("Number of age classes", min_value=2, max_value=10, value=3)
    fertility = [st.sidebar.number_input(f"f_{i}", value=0.5) for i in range(n)]
    survival = [st.sidebar.number_input(f"s_{i}", value=0.8) for i in range(n-1)]
    N0_vec = [st.sidebar.number_input(f"N0_{i}", value=10.0) for i in range(n)]
elif model == "Stochastic":
    repeats = st.sidebar.number_input("Number of repeats", min_value=1, max_value=500, value=100)
    sigma = st.sidebar.slider("Noise sigma", min_value=0.0, max_value=1.0, value=0.1)
    base_model = st.sidebar.selectbox("Base model:", ["Logistic", "Ricker"])

# Utility: plot + export
def plot_and_export(data, title):
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set_title(title)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Population size')
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    st.download_button(_["download_plot"], data=buf, file_name=f"{title}.png", mime="image/png")

# ==== Simulation trigger ====
if st.sidebar.button(_["simulate_btn"]):
    if model == "Logistic Growth":
        traj = simulate_logistic(common['N0'], common['r'], common['K'], T)
        st.line_chart(traj)
        plot_and_export(traj, 'logistic_growth')

    elif model == "Ricker Model":
        traj = simulate_ricker(common['N0'], common['r'], common['K'], T)
        st.line_chart(traj)
        plot_and_export(traj, 'ricker_model')

    elif model == "Delay Model":
        traj = simulate_delay(common['N0'], common['r'], common['K'], T, tau)
        st.line_chart(traj)
        plot_and_export(traj, 'delay_model')

    elif model == "Leslie Matrix":
        history = simulate_leslie(N0_vec, fertility, survival, T)
        df = pd.DataFrame(history, columns=[f"Age {i}" for i in range(len(N0_vec))])
        st.line_chart(df)
        plot_and_export(df.values, 'leslie_matrix')
        L = np.zeros((n, n))
        L[0, :] = fertility
        for i in range(1, n):
            L[i, i-1] = survival[i-1]
        lambda_val = np.max(np.real(np.linalg.eigvals(L)))
        st.write(f"Dominant eigenvalue λ = {lambda_val:.3f}")
        st.download_button(_["download_csv"], data=df.to_csv(index=False).encode('utf-8'), file_name='leslie_matrix.csv')

    elif model == "Stochastic":
        base = simulate_ricker if base_model == 'Ricker' else simulate_logistic
        results = simulate_stochastic(base, common['N0'], common['r'], common['K'], T, sigma=sigma, repeats=repeats)
        mean_traj = results.mean(axis=0)
        st.line_chart(pd.DataFrame(results.T))
        st.write(_["mean_trajectory"])
        st.line_chart(mean_traj)
        plot_and_export(mean_traj, 'stochastic_mean')

# ==== Footer ====
st.sidebar.markdown("---")
st.sidebar.info(_["footer"])
