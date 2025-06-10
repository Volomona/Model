import io
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pdfkit

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

def generate_pdf_report(model_name: str, ts: np.ndarray):
    html = f"""
    <h1>Отчёт по модели: {model_name}</h1>
    <p>Первые 10 значений:</p>
    <pre>{ts[:10]}</pre>
    <p>Размерность массива: {ts.shape}</p>
    """
    output_path = "population_report.pdf"
    pdfkit.from_string(html, output_path)
    return output_path

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
