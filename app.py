# -------------------------------
# Streamlit UI
# -------------------------------
import streamlit as st
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import g4f
from sensitivity import sensitivity_heatmap
from analysis import analyze_behavior, optimize_parameters
from report import generate_pdf_report

# Проверка наличия Streamlit
try:
    import streamlit as st
except ModuleNotFoundError:
    sys.exit("Error: Streamlit is not available. Please install and run locally: `streamlit run app.py`.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройка страницы
st.set_page_config(page_title="Population Dynamics Simulator", layout="wide")
st.title("🌱 Симулятор Популяционной Динамики с Анализом")

# Функции симуляции
# ... (здесь ваши simulate_logistic, simulate_ricker, simulate_leslie, simulate_delay, simulate_stochastic)

# Словарь моделей
models = {
    "Логистический рост": simulate_logistic,
    "Модель Рикера": simulate_ricker,
    "Модель Лесли": lambda N0, r, K, T: simulate_leslie(N0_vec, fertility, survival, T),
    "Модель с задержкой": simulate_delay,
    "Стохастическая симуляция": simulate_stochastic
}

# Sidebar: параметры
model_name = st.sidebar.selectbox("Выберите модель:", list(models.keys()))
T = st.sidebar.slider("Шаги времени (T)", 10, 500, 100, key='T')
N0 = st.sidebar.number_input("Начальная популяция N0", 0.0, 1000.0, 10.0, key='N0')
r = st.sidebar.number_input("Темп роста r", 0.0, 5.0, 0.5, key='r')
K = st.sidebar.number_input("Емкость среды K", 1.0, 1000.0, 100.0, key='K')

# Дополнительные параметры
st.sidebar.subheader("Доп. параметры модели")
if model_name == "Модель с задержкой":
    tau = st.sidebar.number_input("Задержка τ", 1, 10, 2, key='tau')
elif model_name == "Стохастическая симуляция":
    sigma = st.sidebar.number_input("σ (для стохастической)", 0.0, 1.0, 0.1, key='sigma')
    repeats = st.sidebar.number_input("Повторения (для стохастической)", 1, 200, 50, key='repeats')
elif model_name == "Модель Лесли":
    n = st.sidebar.number_input("Число возрастных классов", 2, 10, 3, key='n')
    fertility = [st.sidebar.number_input(f"f_{i}", 0.0, 1.0, 0.5, key=f'fert_{i}') for i in range(n)]
    survival = [st.sidebar.number_input(f"s_{i}", 0.0, 1.0, 0.8, key=f'surv_{i}') for i in range(n-1)]
    N0_vec = [st.sidebar.number_input(f"N0_{i}", 0.0, 1000.0, 10.0, key=f'N0_{i}_vec') for i in range(n)]

# Загрузка CSV для оптимизации
uploaded = st.sidebar.file_uploader("CSV для подгонки параметров", type=["csv"], key='uploader')

# Кнопка симуляции
if st.sidebar.button("Симулировать", key='simulate'):
    # Определяем args в зависимости от модели
    if model_name == "Модель с задержкой":
        ts = models[model_name](N0, r, K, T, tau)
    elif model_name == "Стохастическая симуляция":
        ts = models[model_name](simulate_logistic, N0, r, K, T, sigma=sigma, repeats=repeats)
    elif model_name == "Модель Лесли":
        ts = models[model_name](N0, r, K, T)
    else:
        ts = models[model_name](N0, r, K, T)
    # Сохраняем в сессию
    st.session_state.ts = ts
    st.session_state.model = model_name

# Отображение результатов
if 'ts' in st.session_state:
    ts = st.session_state.ts
    st.subheader(f"Результаты: {st.session_state.model}")
    # График
    fig, ax = plt.subplots()
    ax.plot(ts if ts.ndim==1 else ts)
    st.pyplot(fig)

    # Классификация режима
    mode = analyze_behavior(ts.flatten())
    st.write(f"Режим поведения: {mode}")

    # Кнопка скачивания CSV
    csv = pd.DataFrame(ts).to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Скачать данные CSV",
        data=csv,
        file_name=f"{st.session_state.model}.csv",
        mime="text/csv",
        key='download_csv'
    )

    # GPT-4 анализ
    if st.sidebar.button("Запросить анализ GPT-4", key='gpt4_analysis'):
        snippet = str(ts.flatten()[:10]) + "..." if ts.size > 10 else str(ts.flatten())
        messages = [
            {"role": "user", "content": (
                f"Вы - научный сотрудник. Проанализируйте результаты симуляции.\n"
                f"Тип модели: {st.session_state.model}\n"
                f"Данные (первые 10 точек): {snippet}"
            )}
        ]
        response = g4f.ChatCompletion.create(model=g4f.models.gpt_4, messages=messages)
        st.subheader("Анализ данных GPT-4:")
        st.write(response)

    # Анализ чувствительности
    if st.sidebar.checkbox("Показать анализ чувствительности", key='sens'):
        fig_sens = sensitivity_heatmap(
            models[st.session_state.model],
            {'r': (0.1,1.0,20), 'K': (50,200,20)},
            {'N0': N0, 'r': r, 'K': K},
            T
        )
        st.subheader("Чувствительность (амплитуда)")
        st.pyplot(fig_sens)

    # Подгонка параметров
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        data = df.iloc[:,1].values if df.shape[1]>1 else df.iloc[:,0].values
        res = optimize_parameters(
            models[st.session_state.model], data,
            initial_guess=[N0, r, K],
            bounds=[(0,None),(0,None),(0,None)],
            T=T
        )
        st.subheader("Подгонка параметров")
        st.write(f"Оптимальные: N0={res.x[0]:.2f}, r={res.x[1]:.2f}, K={res.x[2]:.2f}")

    # Скачать PDF отчёт
    if st.sidebar.button("Скачать PDF отчёт", key='pdf_report'):
        path = generate_pdf_report(st.session_state.model, ts)
        st.success(f"Отчёт сохранён: {path}")

st.sidebar.markdown("---")
st.sidebar.info("Разработано Лией Ахметовой")
