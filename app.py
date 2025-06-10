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

# ... (other simulate_* functions unchanged) ...

def simulate_ricker(N0: float, r: float, K: float, T: int) -> np.ndarray:
    Ns = [N0]
    for _ in range(T):
        Ns.append(Ns[-1] * np.exp(r * (1 - Ns[-1] / K)))
    return np.array(Ns)

# -------------------------------
# Analysis functions
# -------------------------------
def analyze_behavior(time_series: np.ndarray) -> str:
    """
    Автоматически определяет режим: стационарность, периодичность или хаос.
    """
    std = np.std(time_series[-int(len(time_series)/2):])
    if std < 1e-3:
        return "Стационарность"
    # простой критерий: значимые колебания
    peaks = np.sum(np.diff(np.sign(np.diff(time_series))) < 0)
    if peaks > 5:
        return "Периодические колебания"
    return "Хаос"


def sensitivity_heatmap(model_func, param_ranges: dict, fixed_args: dict, T: int):
    """
    Строит тепловую карту амплитуды от двух параметров.
    param_ranges: {'r': (0.1,1.0,10), 'K': (50,200,10)}
    """
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
    """
    Подгонка параметров модели по MSE.
    initial_guess: [N0, r, K]
    bounds: [(0, None), (0, None), (0, None)]
    """
    def loss(params):
        sim = model_func(params[0], params[1], params[2], T)
        return np.mean((sim - data)**2)
    res = minimize(loss, initial_guess, bounds=bounds)
    return res


def generate_pdf_report(html_content: str, output_path: str = "report.pdf"):
    """
    Генерирует PDF из HTML через pdfkit.
    """
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
    # ... добавить другие модели ...
}
model_name = st.sidebar.selectbox("Выберите модель:", list(models.keys()))
model_func = models[model_name]

T = st.sidebar.slider("Шаги времени (T)", 10, 500, 100)

# Параметры для анализа чувствительности
if st.sidebar.checkbox("Показать анализ чувствительности"):
    st.sidebar.markdown("#### Настройка анализа")
    p1 = st.sidebar.selectbox("Параметр 1", ["r", "K"])
    p2 = st.sidebar.selectbox("Параметр 2", ["r", "K"])  
    v1_range = st.sidebar.slider(f"Диапазон {p1}", 0.0, 2.0, (0.1,1.0), 0.1)
    v2_range = st.sidebar.slider(f"Диапазон {p2}", 10.0, 500.0, (50.0,200.0), 10.0)

# Основные параметры
N0 = st.sidebar.number_input("Начальная популяция N0", 0.0, 1000.0, 10.0)
r = st.sidebar.number_input("Темп роста r", 0.0, 5.0, 0.5)
K = st.sidebar.number_input("Емкость среды K", 1.0, 1000.0, 100.0)

# Загрузка данных для подгонки
uploaded = st.sidebar.file_uploader("Загрузить CSV для подгонки параметров", type=["csv"])

if st.sidebar.button("Запустить симуляцию"):
    ts = model_func(N0, r, K, T)
    behavior = analyze_behavior(ts)
    st.subheader("Результаты симуляции")
    st.line_chart(pd.DataFrame(ts, columns=["Популяция"]))
    st.write(f"Определённый режим поведения: {behavior}")

    # Анализ чувствительности
    if 'Показать анализ чувствительности' in st.session_state:
        fig = sensitivity_heatmap(
            model_func,
            {p1: (v1_range[0], v1_range[1], 20), p2: (v2_range[0], v2_range[1], 20)},
            {'N0': N0, 'r': r, 'K': K},
            T
        )
        st.subheader("Анализ чувствительности (амплитуда)")
        st.pyplot(fig)

    # Подгонка параметров
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        data = df.iloc[:,1].values
        res = optimize_parameters(model_func, data, [N0, r, K], [(0,None),(0,None),(0,None)], T)
        st.subheader("Подгонка параметров")
        st.write(f"Оптимальные параметры: N0={res.x[0]:.3f}, r={res.x[1]:.3f}, K={res.x[2]:.3f}")

    # Генерация отчёта
    if st.button("Скачать PDF отчёт"):
        html = st.experimental_get_query_params()  # упрощённый пример сборки отчёта
        path = generate_pdf_report(html)
        st.success(f"Отчёт сохранён: {path}")

# Конец кода
