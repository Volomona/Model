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
def simulate_hybrid(N0_vec, T, fert_base, surv_base, K, r_fert, r_surv,
                    delay_fert, delay_surv, migration_rates, env_effect, stoch_intensity):
    n = len(N0_vec)
    buffer_size = max(max(delay_fert), max(delay_surv)) + 1
    buffer = [np.array(N0_vec, dtype=float)] * buffer_size
    history = []
    for t in range(T):
        N_new = np.zeros(n)
        total = buffer[-1].sum()
        noise = np.random.normal(0, stoch_intensity * np.sqrt(buffer[-1] + 1))
        env_f = 1 + env_effect * np.sin(0.1 * t)
        # Рождаемость
        for i in range(n):
            dens = np.exp(-r_fert * total / K)
            N_new[0] += fert_base[i] * dens * env_f * buffer[-1][i]
        # Выживаемость
        for i in range(1, n):
            dens = np.exp(-r_surv * buffer[-delay_surv[i-1]][i-1] / (K/n))
            N_new[i] += surv_base[i-1] * dens * env_f * buffer[-1][i-1]
        # Миграция
        mig = np.zeros(n)
        for i in range(n):
            out = buffer[-1][i] * migration_rates[i]
            mig[i] -= out
            mig += out/(n-1)
        N_new = np.clip(N_new + mig + noise, 0, None)
        buffer.append(N_new)
        if len(buffer) > buffer_size:
            buffer.pop(0)
        history.append(N_new.copy())
    return np.array(history)

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
    hist = [N.copy()]
    L = np.zeros((n,n))
    L[0,:] = fertility
    for i in range(1,n):
        L[i,i-1] = survival[i-1]
    for _ in range(T):
        N = L.dot(N)
        hist.append(N.copy())
    return np.array(hist)

def simulate_delay(N0, r, K, T, tau):
    Ns = [N0] * (tau+1)
    for t in range(tau, T+tau):
        Ns.append(Ns[t] * np.exp(r * (1 - Ns[t-tau] / K)))
    return np.array(Ns[:T+1])

def simulate_stochastic(base_sim, N0, r, K, T, sigma, repeats):
    runs = []
    prog = st.progress(0)
    for i in range(repeats):
        traj = base_sim(N0, r, K, T)
        noise = np.random.normal(0, sigma, size=traj.shape)
        runs.append(np.clip(traj + noise, 0, None))
        prog.progress((i+1)/repeats)
    return np.array(runs)

# -------------------------------
# Analysis functions
# -------------------------------
def analyze_behavior(ts):
    std = np.std(ts[-len(ts)//2:])
    if std < 1e-3:
        return "Стационарность"
    peaks = np.sum(np.diff(np.sign(np.diff(ts))) < 0)
    if peaks > 5:
        return "Периодические колебания"
    return "Хаос"

def sensitivity_heatmap(model, param_ranges, fixed, T):
    p1, p2 = list(param_ranges)
    v1 = np.linspace(*param_ranges[p1])
    v2 = np.linspace(*param_ranges[p2])
    M = np.zeros((len(v1), len(v2)))
    for i, x in enumerate(v1):
        for j, y in enumerate(v2):
            args = fixed.copy()
            args[p1], args[p2] = x, y
            ts = model(*args.values(), T)
            M[i,j] = ts.max() - ts.min()
    fig, ax = plt.subplots()
    c = ax.pcolormesh(v1, v2, M.T, shading='auto')
    fig.colorbar(c, ax=ax)
    ax.set_xlabel(p1)
    ax.set_ylabel(p2)
    return fig

def optimize_parameters(model, data, guess, bounds, T):
    def loss(p):
        return np.mean((model(p[0], p[1], p[2], T) - data)**2)
    return minimize(loss, guess, bounds=bounds)

def generate_pdf_report(model_name, ts):
    html = f"<h1>Отчет по модели {model_name}</h1><pre>{ts[:10]}</pre>"
    path = "report.pdf"
    pdfkit.from_string(html, path)
    return path

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Population Dynamics Simulator", layout="wide")
st.title("🌱 Симулятор популяционной динамики")

models = {
    "Гибридная": simulate_hybrid,
    "Логистический": simulate_logistic,
    "Рикер": simulate_ricker,
    "Лесли": simulate_leslie,
    "С задержкой": simulate_delay,
    "Стохастическая": simulate_stochastic
}
model = st.sidebar.selectbox("Модель:", list(models.keys()))
T = st.sidebar.slider("T", 10, 500, 100)

# Input collection
if model == "Гибридная":
    n = st.sidebar.number_input("классов", 2, 10, 3)
    N0v = [st.sidebar.number_input(f"N0_{i}", 0.0, 1000.0, 10.0) for i in range(n)]
    fert = [st.sidebar.number_input(f"f_{i}", 0.0, 1.0, 0.5) for i in range(n)]
    surv = [st.sidebar.number_input(f"s_{i}", 0.0, 1.0, 0.8) for i in range(n-1)]
    df = [st.sidebar.number_input(f"df_{i}", 0, 5, 1) for i in range(n)]
    ds = [st.sidebar.number_input(f"ds_{i}", 0, 5, 1) for i in range(n-1)]
    mig = [st.sidebar.number_input(f"m_{i}", 0.0, 0.5, 0.1) for i in range(n)]
    K = st.sidebar.number_input("K", 1.0, 1000.0, 100.0)
    rf = st.sidebar.number_input("rf", 0.0, 1.0, 0.1)
    rs = st.sidebar.number_input("rs", 0.0, 1.0, 0.05)
    ee = st.sidebar.slider("env", -1.0, 1.0, 0.2)
    si = st.sidebar.slider("sto", 0.0, 1.0, 0.1)
    if st.sidebar.button("Sim"):
        res = models[model](N0v, T, fert, surv, K, rf, rs, df, ds, mig, ee, si)

elif model in ["Логистический", "Рикер"]:
    N0 = st.sidebar.number_input("N0", 0.0, 1000.0, 10.0)
    r = st.sidebar.number_input("r", 0.0, 5.0, 0.5)
    K = st.sidebar.number_input("K", 1.0, 1000.0, 100.0)
    if st.sidebar.button("Sim"):
        res = models[model](N0, r, K, T)

elif model == "С задержкой":
    N0 = st.sidebar.number_input("N0", 0.0, 1000.0, 10.0)
    r = st.sidebar.number_input("r", 0.0, 5.0, 0.5)
    K = st.sidebar.number_input("K", 1.0, 1000.0, 100.0)
    tau = st.sidebar.number_input("tau", 1, 10, 1)
    if st.sidebar.button("Sim"):
        res = models[model](N0, r, K, T, tau)

elif model == "Лесли":
    n = st.sidebar.number_input("классов", 2, 10, 3)
    N0v = [st.sidebar.number_input(f"N0_{i}", 0.0, 1000.0, 10.0) for i in range(n)]
    fert = [st.sidebar.number_input(f"f_{i}", 0.0, 1.0, 0.5) for i in range(n)]
    surv = [st.sidebar.number_input(f"s_{i}", 0.0, 1.0, 0.8) for i in range(n-1)]
    if st.sidebar.button("Sim"):
        res = models[model](N0v, fert, surv, T)

else:
    N0 = st.sidebar.number_input("N0", 0.0, 1000.0, 10.0)
    r = st.sidebar.number_input("r", 0.0, 5.0, 0.5)
    K = st.sidebar.number_input("K", 1.0, 1000.0, 100.0)
    sigma = st.sidebar.number_input("sigma", 0.0, 1.0, 0.1)
    repeats = st.sidebar.number_input("repeats", 1, 200, 50)
    if st.sidebar.button("Sim"):
        res = models[model](simulate_logistic, N0, r, K, T, sigma, repeats)

# Display results
if 'res' in locals():
    st.subheader(f"Результаты: {model}")
    st.line_chart(pd.DataFrame(res))
    st.write("Режим:", analyze_behavior(res.flatten()))
    if st.sidebar.button("Скачать CSV"):
        export_csv(res, model, str(res[:10]), "")
    try:
        fig = sensitivity_heatmap(
            models[model],
            {'r': (0.1, 1, 20), 'K': (10, 200, 20)},
            {'N0': locals().get('N0'), 'r': locals().get('r'), 'K': locals().get('K')},
            T
        )
        st.pyplot(fig)
    except Exception:
        st.warning("Невозможно построить тепловую карту для этой модели.")
    if st.sidebar.checkbox("Оптимизация параметров"):
        # Здесь можно реализовать загрузку CSV и вызов optimize_parameters
        pass
    if st.sidebar.button("PDF отчёт"):
        path = generate_pdf_report(model, res)
        st.success(f"Отчёт сохранён: {path}")

st.sidebar.info("Разработано Лией Ахметовой")
