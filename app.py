# --- часть кода до этого не меняется (импорты, simulate_* и export_csv) ---

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

num_configs = 1
if model in ["Логистический рост", "Модель Рикера", "Модель с задержкой", "Стохастическая симуляция"]:
    num_configs = st.sidebar.slider("Число конфигураций для сравнения", min_value=1, max_value=5, value=2)

params_list = []

if model == "Модель Лесли":
    n = st.sidebar.number_input("Число возрастных классов", min_value=2, max_value=10, value=3)
    with st.sidebar.expander("Коэффициенты рождаемости (f_i)"):
        fertility = [st.number_input(f"f_{i}", min_value=0.0, value=0.5) for i in range(n)]
    with st.sidebar.expander("Вероятности выживания (s_i)"):
        survival = [st.number_input(f"s_{i}", min_value=0.0, max_value=1.0, value=0.8) for i in range(n-1)]
    with st.sidebar.expander("Начальная популяция по возрастным классам"):
        N0_vec = [st.number_input(f"N0_{i}", min_value=0.0, value=10.0) for i in range(n)]

else:
    for i in range(num_configs):
        with st.sidebar.expander(f"Конфигурация {i + 1}"):
            N0 = st.number_input(f"N0 (конфигурация {i + 1})", min_value=0.0, value=10.0, key=f"N0_{i}")
            r = st.number_input(f"r (конфигурация {i + 1})", min_value=0.0, value=0.1, key=f"r_{i}")
            K = st.number_input(f"K (конфигурация {i + 1})", min_value=1.0, value=100.0, key=f"K_{i}")
            tau = 1
            sigma = 0.1
            repeats = 100
            base_model = "Logistic"
            if model == "Модель с задержкой":
                tau = st.slider(f"τ (конфигурация {i + 1})", min_value=1, max_value=10, value=1, key=f"tau_{i}")
            if model == "Стохастическая симуляция":
                sigma = st.slider(f"Шум σ (конфигурация {i + 1})", min_value=0.0, max_value=1.0, value=0.1, key=f"sigma_{i}")
                repeats = st.number_input(f"Повторений (конфигурация {i + 1})", min_value=1, max_value=200, value=100, key=f"repeats_{i}")
                base_model = st.selectbox(f"Базовая модель (конфигурация {i + 1})", ["Logistic", "Ricker"], key=f"base_model_{i}")
            params_list.append({
                "N0": N0, "r": r, "K": K, "tau": tau,
                "sigma": sigma, "repeats": repeats,
                "base_model": base_model
            })

if st.sidebar.button("Симулировать"):
    with st.spinner("Симуляция..."):
        if model == "Модель Лесли":
            history = simulate_leslie(N0_vec, fertility, survival, T)
            df = pd.DataFrame(history, columns=[f"Возраст {i}" for i in range(n)])
            st.subheader("Модель Лесли")
            st.line_chart(df)
            L = np.zeros((n, n)); L[0, :] = fertility
            for i in range(1, n): L[i, i-1] = survival[i-1]
            lambda_val = np.max(np.real(np.linalg.eigvals(L)))
            export_csv(history, 'leslie_matrix_data')
            st.write(f"Доминирующее собственное значение λ = {lambda_val:.3f}")

        else:
            st.subheader(f"{model} — сравнение конфигураций")
            all_trajectories = []

            for idx, config in enumerate(params_list):
                if model == "Логистический рост":
                    traj = simulate_logistic(config["N0"], config["r"], config["K"], T)
                elif model == "Модель Рикера":
                    traj = simulate_ricker(config["N0"], config["r"], config["K"], T)
                elif model == "Модель с задержкой":
                    traj = simulate_delay(config["N0"], config["r"], config["K"], T, config["tau"])
                elif model == "Стохастическая симуляция":
                    base_sim = simulate_ricker if config["base_model"] == 'Ricker' else simulate_logistic
                    runs = simulate_stochastic(base_sim, config["N0"], config["r"], config["K"], T,
                                               sigma=config["sigma"], repeats=config["repeats"])
                    traj = runs.mean(axis=0)
                all_trajectories.append(traj)

            df_compare = pd.DataFrame({f"Конфигурация {i+1}": traj[:T+1] for i, traj in enumerate(all_trajectories)})
            st.line_chart(df_compare)
            export_csv(df_compare, f"{model.replace(' ', '_')}_compare_data")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Разработано Лией Ахметовой — v2.0 (сравнение параметров)")
