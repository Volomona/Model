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

common = {}
if model != "Модель Лесли":
    common['N0'] = st.sidebar.number_input("Начальная популяция N0", min_value=0.0, value=10.0)
    common['r'] = st.sidebar.number_input("Темп роста r", min_value=0.0, value=0.1)
    common['K'] = st.sidebar.number_input("Емкость K", min_value=1.0, value=100.0)

if model == "Модель с задержкой":
    tau = st.sidebar.slider("Задержка (τ)", min_value=1, max_value=10, value=1)

elif model == "Модель Лесли":
    n = st.sidebar.number_input("Число возрастных классов", min_value=2, max_value=10, value=3)
    with st.sidebar.expander("Коэффициенты рождаемости (f_i)"):
        fertility = [st.number_input(f"f_{i}", min_value=0.0, value=0.5) for i in range(n)]
    with st.sidebar.expander("Вероятности выживания (s_i)"):
        survival = [st.number_input(f"s_{i}", min_value=0.0, max_value=1.0, value=0.8) for i in range(n-1)]
    with st.sidebar.expander("Начальная популяция по возрастным классам"):
        N0_vec = [st.number_input(f"N0_{i}", min_value=0.0, value=10.0) for i in range(n)]

elif model == "Стохастическая симуляция":
    repeats = st.sidebar.number_input("Число повторений", min_value=1, max_value=200, value=100)
    sigma = st.sidebar.slider("Шум (σ)", min_value=0.0, max_value=1.0, value=0.1)
    base_model = st.sidebar.selectbox("Основная модель:", ["Logistic", "Ricker"])

if st.sidebar.button("Симулировать"):
    with st.spinner("Симуляция..."):
        if model == "Логистический рост":
            traj = simulate_logistic(common['N0'], common['r'], common['K'], T)
            st.subheader("Логистический рост")
            st.line_chart(traj)
            export_csv(traj, 'logistic_growth_data')

        elif model == "Модель Рикера":
            traj = simulate_ricker(common['N0'], common['r'], common['K'], T)
            st.subheader("Модель Рикера")
            st.line_chart(traj)
            export_csv(traj, 'ricker_model_data')

        elif model == "Модель с задержкой":
            traj = simulate_delay(common['N0'], common['r'], common['K'], T, tau)
            st.subheader("Модель с задержкой")
            st.line_chart(traj)
            export_csv(traj, 'delay_model_data')

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
            st.download_button("Скачать CSV данных", data=df.to_csv(index=False).encode('utf-8'),
                               file_name='leslie_matrix.csv')

        elif model == "Стохастическая симуляция":
            base_sim = simulate_ricker if base_model == 'Ricker' else simulate_logistic
            results = simulate_stochastic(base_sim, common['N0'], common['r'], common['K'], T,
                                          sigma=sigma, repeats=repeats)
            st.subheader("Стохастическая симуляция")
            st.line_chart(pd.DataFrame(results.T))
            st.write("Средняя траектория:")
            mean_traj = results.mean(axis=0)
            st.line_chart(mean_traj)
            export_csv(results, 'stochastic_simulation_data')


# Footer
st.sidebar.markdown("---")
st.sidebar.info("Разработано Лией Ахметовой — v1.0")
