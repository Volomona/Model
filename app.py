```python
import sys
import io
import logging
import time
from threading import Thread
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def amplitude_of_dynamics(values):
    return np.max(values) - np.min(values)

def generate_heatmap(model_func, param1, param2, param_ranges, fixed_params, steps=20):
    p1_vals = np.linspace(*param_ranges[param1], steps)
    p2_vals = np.linspace(*param_ranges[param2], steps)
    
    amplitudes = np.zeros((steps, steps))

    for i, p1 in enumerate(p1_vals):
        for j, p2 in enumerate(p2_vals):
            params = fixed_params.copy()
            params[param1] = p1
            params[param2] = p2
            pop = model_func(params, steps=300)
            amplitudes[j, i] = amplitude_of_dynamics(pop[-100:])  # последние 100 шагов

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(amplitudes, xticklabels=np.round(p1_vals, 2),
                yticklabels=np.round(p2_vals, 2), cmap="viridis", ax=ax)
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    ax.set_title("Амплитуда колебаний")
    plt.xticks(rotation=45)
    st.pyplot(fig)

def simulate_unified_hybrid(params, timesteps):
    """Объединённая гибридная модель с настраиваемыми параметрами"""
    N0_vec = params["N0_vec"]
    fert_base = params["fert_base"]
    surv_base = params["surv_base"]
    K = params["K"]
    r_fert = params["r_fert"]
    r_surv = params["r_surv"]
    delay_fert = params["delay_fert"]
    delay_surv = params["delay_surv"]
    migration_rates = params["migration_rates"]
    env_effect = params["env_effect"]
    stoch_intensity = params["stoch_intensity"]
    
    r = params["r"]
    m = params["m"]
    immigration = params["immigration"]
    delay = int(params["delay"]) if params["delay"] is not None else 0
    noise_std = params["noise_std"]
    
    use_age_structure = params["use_age_structure"]
    use_density_dependence = params["use_density_dependence"]
    use_migration = params["use_migration"]
    use_noise = params["use_noise"]
    use_delay = params["use_delay"]
    use_env_effect = params["use_env_effect"]
    
    if use_age_structure:
        n = len(N0_vec)
        N = np.array(N0_vec, dtype=float)
        history = [N.copy()]
        buffer_size = max(max(delay_fert), max(delay_surv)) + 1 if use_delay else 1
        buffer = [N.copy()] * buffer_size

        for t in range(timesteps):
            N_new = np.zeros(n)
            total_pop = sum(buffer[-1])

            noise = np.random.normal(0, stoch_intensity * np.sqrt(buffer[-1] + 1)) if use_noise else 0
            
            env_factor = 1.0 + (env_effect * np.sin(t * 0.1)) if use_env_effect else 1.0

            for i in range(n):
                delayed_pop = buffer[-delay_fert[i]][i] if use_delay else buffer[-1][i]
                density_effect = np.exp(-r_fert * (total_pop / K)) if use_density_dependence else 1.0
                fertility = fert_base[i] * density_effect * env_factor
                N_new[0] += fertility * buffer[-1][i]

            for i in range(1, n):
                delayed_pop = buffer[-delay_surv[i-1]][i-1] if use_delay else buffer[-1][i-1]
                density_effect = np.exp(-r_surv * (delayed_pop / (K/n))) if use_density_dependence else 1.0
                survival = surv_base[i-1] * density_effect * env_factor
                N_new[i] += survival * buffer[-1][i-1]

            if use_migration:
                migration = np.zeros(n)
                for i in range(n):
                    outflow = buffer[-1][i] * migration_rates[i]
                    migration[i] -= outflow
                    for j in range(n):
                        if i != j:
                            migration[j] += outflow / (n-1)
                N_new += migration

            N_new = np.clip(N_new + noise, 0, None)
            buffer.append(N_new)
            if len(buffer) > buffer_size:
                buffer.pop(0)
            history.append(N_new.copy())

        return np.array(history)
    else:
        population = params["N0"]
        history = [population]
        buffer = [population] * delay if use_delay else [population]

        for t in range(1, timesteps):
            prev = buffer[-1] if use_delay else history[-1]

            growth = r * prev * (1 - prev / K) if use_density_dependence else 0
            migration = m * (immigration - population) if use_migration else 0
            noise = np.random.normal(0, noise_std) if use_noise else 0

            population += growth + migration + noise
            population = max(population, 0)
            history.append(population)

            buffer.append(population)
            if len(buffer) > delay:
                buffer.pop(0)

        return np.array(history)

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

def simulate_stochastic(base_sim, *args, sigma: float = 0.1, repeats: int = 100) -> np.ndarray:
    runs = []
    progress = st.progress(0)
    for i in range(repeats):
        traj = base_sim(*args)
        noise = np.random.normal(0, sigma, size=traj.shape)
        runs.append(np.clip(traj + noise, 0, None))
        progress.progress((i + 1) / repeats)
    return np.array(runs)

def export_csv(data, filename, typem, str_data):
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
    try:
        import g4f
        response = g4f.ChatCompletion.create(
            model=g4f.models.gpt_4,
            messages=[{"role": "user", "content": f"Воспринимай график как данные точек. Проанализируй график или возможно несколько графиков популяционной модели. Ничего не проси уточнить. Это не чат, ты пишешь 1 раз и всё. Обязательно форматируй текст по Markdown, будто ты научный сотрудник. Тип модели: {typem}, вот результат симуляции: {str_data}"}],
        )
        container = st.container(border=True)
        container.write("Анализ полученных данных:")
        container.write(response)
    except Exception as e:
        st.warning(f"Не удалось выполнить анализ с помощью g4f: {str(e)}")

st.set_page_config(page_title="Симулятор Популяционной Динамики", layout="wide")
st.title("🌱 Симулятор Популяционной Динамики")

model_info = {
    "Гибридная модель": "Интегративная модель с возрастной структурой, плотностной зависимостью, задержками, стохастичностью и миграцией.",
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

with st.expander("🔬 Анализ чувствительности (тепловая карта амплитуды)"):
    model_type = st.selectbox("Выберите модель", ["Логистическая", "Рикера", "Гибридная"])
    param1 = st.selectbox("Параметр по оси X", ["r", "K"] if model_type != "Гибридная" else ["r_fert", "K"])
    param2 = st.selectbox("Параметр по оси Y", ["r", "K"] if model_type != "Гибридная" else ["r_surv", "K"])
    steps = st.slider("Разбиение сетки", 10, 50, 20)
    run_heatmap = st.button("Построить тепловую карту")

    if run_heatmap:
        param_ranges = {
            "r": (0.1, 3.0),
            "K": (50, 500),
            "r_fert": (0.01, 0.5),
            "r_surv": (0.01, 0.5)
        }
        if model_type == "Логистическая":
            fixed = {"r": 1.5, "K": 300, "N0": 10}
            def wrapper(params, steps=300):
                return simulate_logistic(params["N0"], params["r"], params["K"], steps)
        elif model_type == "Рикера":
            fixed = {"r": 1.5, "K": 300, "N0": 10}
            def wrapper(params, steps=300):
                return simulate_ricker(params["N0"], params["r"], params["K"], steps)
        elif model_type == "Гибридная":
            fixed = {
                "N0_vec": [10, 10, 10], "fert_base": [0.5, 0.5, 0.5], "surv_base": [0.8, 0.8],
                "K": 100, "r_fert": 0.1, "r_surv": 0.05, "delay_fert": [1, 1, 1], "delay_surv": [1, 1],
                "migration_rates": [0.1, 0.1, 0.1], "env_effect": 0.2, "stoch_intensity": 0.1,
                "use_age_structure": True, "use_density_dependence": True, "use_migration": True,
                "use_noise": True, "use_delay": True, "use_env_effect": True,
                "r": None, "m": None, "immigration": None, "delay": 0, "noise_std": None
            }
            def wrapper(params, steps=300):
                return simulate_unified_hybrid(params, steps).sum(axis=1)  # Общая численность
        generate_heatmap(wrapper, param1, param2, param_ranges, fixed, steps)

if model == "Гибридная модель":
    config_params = []
    configs_count = st.sidebar.number_input("Количество конфигураций", min_value=1, max_value=5, value=1)
    for config_idx in range(configs_count):
        with st.sidebar.expander(f"Конфигурация #{config_idx+1}", expanded=config_idx==0):
            st.markdown("**Основные параметры**")
            use_age_structure = st.checkbox("Возрастная структура", value=True, key=f"age_struct_{config_idx}")
            
            if use_age_structure:
                n = st.number_input("Число возрастных групп", min_value=2, max_value=10, value=3, key=f"n_groups_{config_idx}")
                st.markdown("**Начальная численность**")
                N0_vec = [st.number_input(f"🔢 Группа {i+1}", min_value=0.0, value=10.0, key=f"N0_{i}_{config_idx}") for i in range(n)]
                st.markdown("**Рождаемость**")
                fert_base = [st.number_input(f"👶 Группа {i+1}", min_value=0.0, value=0.5, key=f"fert_{i}_{config_idx}") for i in range(n)]
                st.markdown("**Выживаемость**")
                surv_base = [st.number_input(f"🔄 Группа {i+1}", min_value=0.0, max_value=1.0, value=0.8, key=f"surv_{i}_{config_idx}") for i in range(n-1)]
                st.markdown("**Задержка реакции**")
                delay_fert = [st.number_input(f"⏳ Группа {i+1}", min_value=0, max_value=5, value=1, key=f"delay_fert_{i}_{config_idx}") for i in range(n)]
                delay_surv = [st.number_input(f"⏳ Переход {i+1}→{i+2}", min_value=0, max_value=5, value=1, key=f"delay_surv_{i}_{config_idx}") for i in range(n-1)]
                st.markdown("**Миграция**")
                migration_rates = [st.number_input(f"🔄 Группа {i+1}", min_value=0.0, max_value=0.5, value=0.1, key=f"migr_{i}_{config_idx}") for i in range(n)]
            else:
                N0 = st.number_input("Начальная популяция", min_value=0.0, value=10.0, key=f"N0_simple_{config_idx}")
            
            st.markdown("**Общие параметры**")
            use_density_dependence = st.checkbox("Плотностная зависимость", value=True, key=f"density_{config_idx}")
            K = st.number_input("Емкость среды (K)", min_value=1.0, value=100.0, key=f"K_{config_idx}") if use_density_dependence else 100.0
            
            st.markdown("**Параметры роста**")
            if use_age_structure:
                r_fert = st.number_input("Влияние на рождаемость", min_value=0.0, value=0.1, key=f"r_fert_{config_idx}")
                r_surv = st.number_input("Влияние на выживаемость", min_value=0.0, value=0.05, key=f"r_surv_{config_idx}")
            else:
                r = st.number_input("Темп роста (r)", min_value=0.0, value=0.1, key=f"r_{config_idx}")
            
            st.markdown("**Дополнительные факторы**")
            use_migration = st.checkbox("Миграция", value=True, key=f"migr_flag_{config_idx}")
            if use_migration and not use_age_structure:
                m = st.number_input("Коэф. миграции", min_value=0.0, max_value=1.0, value=0.1, key=f"m_{config_idx}")
                immigration = st.number_input("Уровень иммиграции", min_value=0, value=50, key=f"imm_{config_idx}")
            else:
                m = None
                immigration = None
            
            use_noise = st.checkbox("Стохастичность", value=True, key=f"noise_{config_idx}")
            if use_noise:
                if use_age_structure:
                    stoch_intensity = st.number_input("Интенсивность шума", min_value=0.0, max_value=1.0, value=0.1, key=f"stoch_{config_idx}")
                    noise_std = None
                else:
                    noise_std = st.number_input("Ст. отклонение шума", min_value=0.0, max_value=5.0, value=0.5, key=f"noise_std_{config_idx}")
                    stoch_intensity = None
            else:
                stoch_intensity = None
                noise_std = None
            
            use_delay = st.checkbox("Задержка", value=True, key=f"delay_{config_idx}")
            if use_delay and not use_age_structure:
                delay = st.number_input("Величина задержки", min_value=0, max_value=50, value=10, key=f"delay_val_{config_idx}")
            else:
                delay = 0
            
            use_env_effect = st.checkbox("Влияние среды", value=False, key=f"env_{config_idx}")
            if use_env_effect and use_age_structure:
                env_effect = st.slider("Сила влияния", -1.0, 1.0, 0.2, key=f"env_eff_{config_idx}")
            else:
                env_effect = None
            
            params = {
                "use_age_structure": use_age_structure,
                "N0": N0 if not use_age_structure else None,
                "N0_vec": N0_vec if use_age_structure else None,
                "fert_base": fert_base if use_age_structure else None,
                "surv_base": surv_base if use_age_structure else None,
                "K": K,
                "r": r if not use_age_structure else None,
                "r_fert": r_fert if use_age_structure else None,
                "r_surv": r_surv if use_age_structure else None,
                "delay_fert": delay_fert if use_age_structure else None,
                "delay_surv": delay_surv if use_age_structure else None,
                "migration_rates": migration_rates if use_age_structure else None,
                "m": m,
                "immigration": immigration,
                "delay": delay,
                "noise_std": noise_std,
                "stoch_intensity": stoch_intensity,
                "env_effect": env_effect,
                "use_density_dependence": use_density_dependence,
                "use_migration": use_migration,
                "use_noise": use_noise,
                "use_delay": use_delay,
                "use_env_effect": use_env_effect
            }
            config_params.append(params)

elif model == "Модель с задержкой":
    tau_values = st.sidebar.multiselect(
        "Значения задержки (τ)",
        options=list(range(1, 11)),
        default=[1, 2]
    )
    common = {
        'N0': st.sidebar.number_input("Начальная популяция N0", min_value=0.0, value=10.0),
        'r': st.sidebar.number_input("Темп роста r", min_value=0.0, value=0.1),
        'K': st.sidebar.number_input("Емкость K", min_value=1.0, value=100.0)
    }

elif model == "Модель Лесли":
    n = st.sidebar.number_input("Число возрастных классов", min_value=2, max_value=10, value=3)
    with st.sidebar.expander("Коэффициенты рождаемости (f_i)"):
        fertility = [st.number_input(f"f_{i}", min_value=0.0, value=0.5) for i in range(n)]
    with st.sidebar.expander("Вероятности выживания (s_i)"):
        survival = [st.number_input(f"s_{i}", min_value=0.0, max_value=1.0, value=0.8) for i in range(n-1)]
    with st.sidebar.expander("Начальная популяция по возрастным классам"):
        N0_vec = [st.number_input(f"N0_{i}", min_value=0.0, value=10.0) for i in range(n)]

elif model == "Стохастическая симуляция":
    repeats = st.sidebar.number_input("Число повторений", min_value=1, max_value=200, value=50)
    sigma_values = st.sidebar.multiselect(
        "Значения шума (σ)",
        options=[0.0, 0.05, 0.1, 0.2, 0.5],
        default=[0.1]
    )
    base_model = st.sidebar.selectbox("Основная модель:", ["Логистический рост", "Модель Рикера"])
    base_sim = simulate_logistic if base_model == "Логистический рост" else simulate_ricker
    common = {
        'N0': st.sidebar.number_input("Начальная популяция N0", min_value=0.0, value=10.0),
        'r': st.sidebar.number_input("Темп роста r", min_value=0.0, value=0.1),
        'K': st.sidebar.number_input("Емкость K", min_value=1.0, value=100.0)
    }

else:
    configs_count = st.sidebar.number_input("Количество конфигураций", min_value=1, max_value=5, value=1)
    config_params = []
    for i in range(configs_count):
        st.sidebar.markdown(f"**Конфигурация #{i+1}**")
        N0_i = st.sidebar.number_input(f"N0 (начальная популяция) #{i+1}", min_value=0.0, value=10.0)
        r_i = st.sidebar.number_input(f"r (темп роста) #{i+1}", min_value=0.0, value=0.1)
        K_i = st.sidebar.number_input(f"K (емкость) #{i+1}", min_value=1.0, value=100.0)
        config_params.append((N0_i, r_i, K_i))

if st.sidebar.button("Симулировать"):
    with st.spinner("Симуляция..."):
        if model == "Гибридная модель":
            all_trajs = {}
            config_descriptions = []
            for idx, params in enumerate(config_params):
                population = simulate_unified_hybrid(params, T)
                if params["use_age_structure"]:
                    df = pd.DataFrame(population, columns=[f"Возраст {i}" for i in range(len(params["N0_vec"]))])
                    st.subheader(f"Конфигурация #{idx+1} - Динамика по возрастным классам")
                    st.line_chart(df)
                    total_pop = df.sum(axis=1)
                    st.subheader(f"Конфигурация #{idx+1} - Общая численность")
                    st.line_chart(pd.DataFrame(total_pop, columns=["Общая численность"]))
                    params_str = (f"Возрастная структура: {len(params['N0_vec'])} групп\n"
                                f"K={params['K']}, r_fert={params['r_fert']}, r_surv={params['r_surv']}\n"
                                f"Факторы: плотность={params['use_density_dependence']}, "
                                f"миграция={params['use_migration']}, шум={params['use_noise']}, "
                                f"задержка={params['use_delay']}, среда={params['use_env_effect']}")
                else:
                    df = pd.DataFrame(population, columns=["Популяция"])
                    st.subheader(f"Конфигурация #{idx+1} - Динамика популяции")
                    st.line_chart(df)
                    params_str = (f"r={params['r']}, K={params['K']}\n"
                                f"Факторы: плотность={params['use_density_dependence']}, "
                                f"миграция={params['use_migration']}, шум={params['use_noise']}, "
                                f"задержка={params['use_delay']}")
                all_trajs[f"Конфигурация #{idx+1}"] = population
                config_descriptions.append(params_str)
            export_csv(all_trajs, 'unified_hybrid', 'Гибридная модель',
                      f"Конфигурации:\n{'\n'.join(config_descriptions)}\nДанные:\n{all_trajs}")

        elif model == "Логистический рост":
            if configs_count == 1:
                traj = simulate_logistic(config_params[0][0], config_params[0][1], config_params[0][2], T)
                df = pd.DataFrame(traj, columns=["Популяция"])
                st.subheader("Логистический рост")
                st.line_chart(df)
                export_csv(df, 'logistic_growth', 'Логистический рост',
                           f"Одна траектория: N0={config_params[0][0]}, r={config_params[0][1]}, K={config_params[0][2]}\nДанные:\n{traj}")
            else:
                all_trajs = {}
                config_descriptions = []
                for idx, (N0_i, r_i, K_i) in enumerate(config_params):
                    traj = simulate_logistic(N0_i, r_i, K_i, T)
                    all_trajs[f"Конфигурация #{idx + 1} (r={r_i}, K={K_i})"] = traj
                    config_descriptions.append(f"Конфигурация #{idx + 1}: N0={N0_i}, r={r_i}, K={K_i}")
                df = pd.DataFrame(all_trajs)
                st.subheader("Логистический рост - Несколько конфигураций")
                st.line_chart(df)
                export_csv(df, 'logistic_growth_multiple', 'Логистический рост',
                           f"Множественные траектории:\n{'\n'.join(config_descriptions)}\nДанные:\n{all_trajs}")

        elif model == "Модель Рикера":
            if configs_count == 1:
                traj = simulate_ricker(config_params[0][0], config_params[0][1], config_params[0][2], T)
                df = pd.DataFrame(traj, columns=["Популяция"])
                st.subheader("Модель Рикера")
                st.line_chart(df)
                export_csv(df, 'ricker_model', 'Модель Рикера',
                           f"Одна траектория: N0={config_params[0][0]}, r={config_params[0][1]}, K={config_params[0][2]}\nДанные:\n{traj}")
            else:
                all_trajs = {}
                config_descriptions = []
                for idx, (N0_i, r_i, K_i) in enumerate(config_params):
                    traj = simulate_ricker(N0_i, r_i, K_i, T)
                    all_trajs[f"Конфигурация #{idx + 1} (r={r_i}, K={K_i})"] = traj
                    config_descriptions.append(f"Конфигурация #{idx + 1}: N0={N0_i}, r={r_i}, K={K_i}")
                df = pd.DataFrame(all_trajs)
                st.subheader("Модель Рикера - Несколько конфигураций")
                st.line_chart(df)
                export_csv(df, 'ricker_model_multiple', 'Модель Рикера',
                           f"Множественные траектории:\n{'\n'.join(config_descriptions)}\nДанные:\n{all_trajs}")

        elif model == "Модель с задержкой":
            if not tau_values:
                st.warning("Выберите хотя бы одно значение τ")
            else:
                all_trajs = {}
                tau_descriptions = []
                for tau_i in tau_values:
                    traj = simulate_delay(common['N0'], common['r'], common['K'], T, tau_i)
                    all_trajs[f"τ = {tau_i}"] = traj
                    tau_descriptions.append(
                        f"Задержка τ={tau_i} при N0={common['N0']}, r={common['r']}, K={common['K']}")
                df = pd.DataFrame(all_trajs)
                st.subheader("Модель с задержкой - Разные τ")
                st.line_chart(df)
                export_csv(df, 'delay_model_multiple_tau', 'Модель с задержкой',
                           f"Траектории с разными задержками:\n{'\n'.join(tau_descriptions)}\nДанные:\n{all_trajs}")

        elif model == "Модель Лесли":
            history = simulate_leslie(N0_vec, fertility, survival, T)
            df = pd.DataFrame(history, columns=[f"Возраст {i}" for i in range(n)])
            st.subheader("Модель Лесли")
            st.line_chart(df)
            L = np.zeros((n, n))
            L[0, :] = fertility
            for i in range(1, n):
                L[i, i - 1] = survival[i - 1]
            lambda_val = np.max(np.real(np.linalg.eigvals(L)))
            st.write(f"Доминирующее собственное значение λ = {lambda_val:.3f}")
            export_csv(df, 'leslie_matrix', 'Модель Лесли', history)

        elif model == "Стохастическая симуляция":
            if not sigma_values:
                st.warning("Выберите хотя бы одно значение σ")
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                all_means = {}
                sigma_descriptions = []
                for sigma in sigma_values:
                    results = simulate_stochastic(
                        base_sim,
                        common['N0'],
                        common['r'],
                        common['K'],
                        T,
                        sigma=sigma,
                        repeats=repeats
                    )
                    for i in range(repeats):
                        ax.plot(results[i], alpha=0.1, linewidth=0.8)
                    mean_traj = results.mean(axis=0)
                    ax.plot(mean_traj, linewidth=2, label=f'σ={sigma}')
                    all_means[f"σ={sigma}"] = mean_traj
                    sigma_descriptions.append(f"σ={sigma} (N0={common['N0']}, r={common['r']}, K={common['K']})")
                ax.set_title(f"Стохастическая симуляция ({repeats} траекторий на сигму)")
                ax.legend()
                st.pyplot(fig)
                means_df = pd.DataFrame(all_means)
                st.subheader("Средние траектории для разных уровней шума")
                st.line_chart(means_df)
                export_csv(means_df, 'stochastic_simulation_means', 'Стохастическая модель',
                           f"Стохастические траектории с параметрами:\n{'\n'.join(sigma_descriptions)}\n"
                           f"Средние значения:\n{all_means}\n"
                           f"Базовые параметры: N0={common['N0']}, r={common['r']}, K={common['K']}")

st.sidebar.markdown("---")
st.sidebar.info("Разработано Лией Ахметовой")
```
