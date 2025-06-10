import sys
import io
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize
from SALib.sample import saltelli
from SALib.analyze import sobol
import pdfkit

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Simulation functions
# -------------------------------
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
        N_next = Ns[t] * np.exp(r * (1 - Ns[t - tau] / K))
        Ns.append(N_next)
    return np.array(Ns[:T + 1])

def simulate_stochastic(base_sim, N0: float, r: float, K: float, T: int, sigma: float = 0.1, repeats: int = 100) -> np.ndarray:
    runs = []
    progress = st.progress(0)
    for i in range(repeats):
        traj = base_sim(N0, r, K, T)
        noise = np.random.normal(0, sigma, size=traj.shape)
        runs.append(np.clip(traj + noise, 0, None))
        progress.progress((i + 1) / repeats)
    return np.array(runs)

# -------------------------------
# Export & GPT4 analysis
# -------------------------------
def export_csv(data, filename, model_type: str, simulation_params: str):
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Скачать данные CSV",
        data=csv,
        file_name=f"{filename}.csv",
        mime="text/csv"
    )
    # GPT-4 аналитика
    import g4f
    snippet = str(data[:10]) + "..." if len(data) > 10 else str(data)
    response = g4f.ChatCompletion.create(
        model=g4f.models.gpt_4,
        messages=[{"role": "user", "content": f"Вы - научный сотрудник. Проанализируйте результаты симуляции."
                    f"\nТип модели: {model_type}"
                    f"\nПараметры: {simulation_params}"
                    f"\nДанные (первые 10 точек): {snippet}"}]
    )
    st.subheader("Анализ данных GPT-4:")
    st.write(response)

# -------------------------------
# Analysis functions
# -------------------------------
def analyze_behavior(time_series: np.ndarray) -> str:
    std = np.std(time_series[-int(len(time_series)/2):])
    if std < 1e-3:
        return "Стационарность"
    peaks = np.sum(np.diff(np.sign(np.diff(time_series))) < 0)
    if peaks > 5:
        return "Периодические колебания"
    return "Хаос"


def sensitivity_heatmap(model_func, param_ranges: dict, fixed_args: dict, T: int):
    p1, p2 = list(param_ranges.keys())
    v1 = np.linspace(*param_ranges[p1])
    v2 = np.linspace(*param_ranges[p2])
    amp = np.zeros((len(v1), len(v2)))
    for i, x in enumerate(v1):
        for j, y in enumerate(v2):
            args = fixed_args.copy()
            args[p1], args[p2] = x, y
            ts = model_func(*args.values(), T)
            amp[i,j] = ts.max() - ts.min()
    fig, ax = plt.subplots()
    c = ax.pcolormesh(v1, v2, amp.T, shading='auto')
    fig.colorbar(c, ax=ax)
    ax.set_xlabel(p1)
    ax.set_ylabel(p2)
    return fig


def optimize_parameters(model_func, data: np.ndarray, initial_guess: list, bounds: list, T: int):
    def loss(params):
        sim = model_func(params[0], params[1], params[2], T)
        return np.mean((sim - data)**2)
    res = minimize(loss, initial_guess, bounds=bounds)
    return res


def generate_pdf_report(html_content: str, output_path: str = "report.pdf"):
    pdfkit.from_string(html_content, output_path)
    return output_path

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Population Dynamics Simulator", layout="wide")
st.title("🌱 Симулятор Популяционной Динамики с Анализом")

# Sidebar: выбор модели и параметров
models = {
    "Логистический рост": simulate_logistic,
    "Модель Рикера": simulate_ricker,
    "Модель Лесли": lambda N0, r, K, T: simulate_leslie(N0, r, K, T),  # адаптация интерфейса
    "Модель с задержкой": lambda N0, r, K, T: simulate_delay(N0, r, K, T, tau),
    "Стохастическая симуляция": lambda N0, r, K, T: simulate_stochastic(simulate_logistic, N0, r, K, T, sigma, repeats)
}
model_name = st.sidebar.selectbox("Выберите модель:", list(models.keys()))

# Загрузка параметров
T = st.sidebar.slider("Шаги времени (T)", 10, 500, 100)
N0 = st.sidebar.number_input("Начальная популяция N0", 0.0, 1000.0, 10.0)
r = st.sidebar.number_input("Темп роста r", 0.0, 5.0, 0.5)
K = st.sidebar.number_input("Емкость среды K", 1.0, 1000.0, 100.0)

# Доп. параметры для конкретных моделей
tau = st.sidebar.number_input("Задержка τ (для модели с задержкой)", 1, 10, 2)
sigma = st.sidebar.number_input("σ (для стохастической)", 0.0, 1.0, 0.1)
repeats = st.sidebar.number_input("Повторения (для стохастической)", 1, 200, 50)

# Параметры Лесли
n = st.sidebar.number_input("Число возрастных классов", 2, 10, 3)
fertility = [st.sidebar.number_input(f"f_{i}", 0.0, 1.0, 0.5) for i in range(n)]
survival = [st.sidebar.number_input(f"s_{i}", 0.0, 1.0, 0.8) for i in range(n-1)]
N0_vec = [st.sidebar.number_input(f"N0_{i}", 0.0, 100.0, 10.0) for i in range(n)]

# File uploader
uploaded = st.sidebar.file_uploader("Загрузить CSV для подгонки параметров", type=["csv"])

if st.sidebar.button("Симулировать"):
    # Запуск выбранной модели
    if model_name in ["Логистический рост", "Модель Рикера"]:
        ts = models[model_name](N0, r, K, T)
    elif model_name == "Модель Лесли":
        ts = simulate_leslie(N0_vec, fertility, survival, T)
    elif model_name == "Модель с задержкой":
        ts = simulate_delay(N0, r, K, T, tau)
    else:
        ts = simulate_stochastic(simulate_logistic, N0, r, K, T, sigma=sigma, repeats=repeats)

    # Визуализация и анализ
    st.subheader(f"Результаты: {model_name}")
    st.line_chart(pd.DataFrame(ts if ts.ndim==1 else ts))
    st.write(f"Режим поведения: {analyze_behavior(ts.flatten())}")

    # Скачать CSV и GPT-анализ
    export_csv(ts, model_name.replace(" ", "_"), model_name, f"N0={N0}, r={r}, K={K}, tau={tau}, sigma={sigma}")

    # Анализ чувствительности
    if st.sidebar.checkbox("Показать анализ чувствительности"):
        fig = sensitivity_heatmap(
            models[model_name],
            {'r': (0.1,1.0,20), 'K': (50,200,20)},
            {'N0': N0, 'r': r, 'K': K},
            T
        )
        st.subheader("Чувствительность (амплитуда)")
        st.pyplot(fig)

    # Подгонка параметров
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        data = df.iloc[:,1].values if df.shape[1]>1 else df.iloc[:,0].values
        res = optimize_parameters(models[model_name], data, [N0, r, K], [(0,None),(0,None),(0,None)], T)
        st.subheader("Подгонка параметров")
        st.write(f"Оптимальные: N0={res.x[0]:.2f}, r={res.x[1]:.2f}, K={res.x[2]:.2f}")

    # Скачать отчёт
    if st.sidebar.button("Скачать PDF отчёт"):
        html = st.experimental_get_query_params()
        path = generate_pdf_report(str(html))
        st.success(f"Отчёт сохранён: {path}")

st.sidebar.markdown("---")
st.sidebar.info("Разработано Лией Ахметовой")
