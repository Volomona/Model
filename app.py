import sys
import io
import logging
import time
from threading import Thread
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import time
try:
    import streamlit as st
except ModuleNotFoundError:
    sys.exit("Error: Streamlit is not available. Please install and run locally: `streamlit run app.py`.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simulate_hybrid(N0_vec: list, T: int,
                    fert_base: list, surv_base: list,
                    K: float, r_fert: float, r_surv: float,
                    delay_fert: list, delay_surv: list,
                    migration_rates: list = None,
                    env_effect: float = 0.0,
                    stoch_intensity: float = 0.1) -> np.ndarray:
    """Гибридная модель с возрастной структурой"""
    n = len(N0_vec)
    N = np.array(N0_vec, dtype=float)
    history = [N.copy()]

    buffer_size = max(max(delay_fert), max(delay_surv)) + 1
    buffer = [N.copy()] * buffer_size

    if migration_rates is None:
        migration_rates = [0.0] * n

    total_pop = np.sum(N)

    for t in range(T):
        N_new = np.zeros(n)
        total_pop = sum(buffer[-1])

        noise = np.random.normal(0, stoch_intensity * np.sqrt(buffer[-1] + 1))
        env_factor = 1.0 + env_effect * np.sin(t * 0.1)

        for i in range(n):
            delayed_pop = buffer[-delay_fert[i]][i]
            density_effect = np.exp(-r_fert * (total_pop / K))
            fertility = fert_base[i] * density_effect * env_factor
            N_new[0] += fertility * buffer[-1][i]

        for i in range(1, n):
            delayed_pop = buffer[-delay_surv[i-1]][i-1]
            density_effect = np.exp(-r_surv * (delayed_pop / (K/n)))
            survival = surv_base[i-1] * density_effect * env_factor
            N_new[i] += survival * buffer[-1][i-1]

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

def simulate_hybrid_model(params, timesteps, use_density_dependence=True, use_migration=True, use_noise=True, use_delay=True):
    """Альтернативная гибридная модель с настраиваемыми факторами"""
    r = params["r"]
    K = params["K"]
    m = params["m"]
    immigration = params["immigration"]
    delay = int(params["delay"])
    noise_std = params["noise_std"]
    population = params["N0"]
    history = [population]

    if use_delay:
        buffer = [population] * delay
    else:
        buffer = [population]

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

def export_csv(data, filename, typem, str):
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
    import g4f

    response = g4f.ChatCompletion.create(
        model=g4f.models.gpt_4,
        messages=[{"role": "user", "content": f"Воспринимай график как данные точек.Проанализируй график или возможно несколько графиков популяционной модели.Ничего не проси уточнить. Это не чат ты пишешь 1 раз и всё.Обязательно форматируй текст по MakrDown. будто ты научный сотрудник. Тип модели:{typem} вот результат симуляции: {str}"}],
    )
    container = st.container(border=True)
    container.write("Анализ полученных данных:")
    container.write(response)

# Основной интерфейс
st.set_page_config(page_title="Population Dynamics Simulator", layout="wide")
st.title("🌱 Симулятор Популяционной Динамики")

model_info = {
    "Гибридная модель": "Интегративная модель с возрастной структурой",
    "Альтернативная гибридная модель": "Упрощенная гибридная модель с настраиваемыми факторами",
    "Логистический рост": "Классическая логистическая карта",
    "Модель Рикера": "Экспоненциальный рост с зависимостью от плотности",
    "Модель Лесли": "Возрастная структура через матрицу Лесли",
    "Модель с задержкой": "Популяция зависит от прошлого состояния",
    "Стохастическая симуляция": "Добавляет гауссов шум к запускам",
}

st.sidebar.info("Выберите модель и установите параметры ниже.")
model = st.sidebar.selectbox("Выберите модель:", list(model_info.keys()))
st.sidebar.caption(model_info[model])

st.sidebar.markdown("### Общие параметры")
T = st.sidebar.number_input("Шаги времени (T)", min_value=1, max_value=500, value=100)

common = {}
if model not in ["Модель Лесли", "Гибридная модель", "Альтернативная гибридная модель"]:
    common['N0'] = st.sidebar.number_input("Начальная популяция N0", min_value=0.0, value=10.0)
    common['r'] = st.sidebar.number_input("Темп роста r", min_value=0.0, value=0.1)
    common['K'] = st.sidebar.number_input("Емкость K", min_value=1.0, value=100.0)

if model == "Модель с задержкой":
    tau_values = st.sidebar.multiselect(
        "Значения задержки (τ)",
        options=list(range(1, 11)),
        default=[1, 2]
    )
elif model == "Гибридная модель":
    n = st.sidebar.number_input("Число возрастных групп", min_value=2, max_value=10, value=3)
    with st.sidebar.expander("Начальная численность"):
        N0_vec = [st.number_input(f"🔢 Группа {i+1}", min_value=0.0, value=10.0) for i in range(n)]
    with st.sidebar.expander("Рождаемость"):
        fert_base = [st.number_input(f"👶 Группа {i+1}", min_value=0.0, value=0.5) for i in range(n)]
    with st.sidebar.expander("Выживаемость"):
        surv_base = [st.number_input(f"🔄 Группа {i+1}", min_value=0.0, max_value=1.0, value=0.8) for i in range(n-1)]
    with st.sidebar.expander("Задержка реакции"):
        delay_fert = [st.number_input(f"⏳ Группа {i+1}", min_value=0, max_value=5, value=1) for i in range(n)]
        delay_surv = [st.number_input(f"⏳ Переход {i+1}→{i+2}", min_value=0, max_value=5, value=1) for i in range(n-1)]
    with st.sidebar.expander("Миграция"):
        migration_rates = [st.number_input(f"🔄 Группа {i+1}", min_value=0.0, max_value=0.5, value=0.1) for i in range(n)]
    
    K = st.sidebar.number_input("📊 Максимальная численность (K)", min_value=1.0, value=100.0)
    r_fert = st.sidebar.number_input("📉 Влияние на рождаемость", min_value=0.0, value=0.1)
    r_surv = st.sidebar.number_input("📉 Влияние на выживаемость", min_value=0.0, value=0.05)
    env_effect = st.sidebar.slider("🌡️ Влияние среды", -1.0, 1.0, 0.2)
    stoch_intensity = st.sidebar.slider("🎲 Случайные колебания", 0.0, 1.0, 0.1)

elif model == "Альтернативная гибридная модель":
    st.sidebar.header("Параметры модели")
    r = st.sidebar.slider("r (темп роста)", 0.0, 4.0, 1.5, 0.1)
    K = st.sidebar.slider("K (ёмкость среды)", 10, 500, 100, 10)
    m = st.sidebar.slider("m (коэффициент миграции)", 0.0, 1.0, 0.1, 0.05)
    immigration = st.sidebar.slider("Иммиграция", 0, 500, 50, 10)
    delay = st.sidebar.slider("Задержка (шаги)", 0, 50, 10)
    noise_std = st.sidebar.slider("Ст. отклонение шума", 0.0, 5.0, 0.5, 0.1)
    N0 = st.sidebar.slider("Начальное население", 0, 500, 10)
    
    st.sidebar.header("Факторы модели")
    use_density_dependence = st.sidebar.checkbox("Плотностная зависимость", value=True)
    use_migration = st.sidebar.checkbox("Миграция", value=True)
    use_noise = st.sidebar.checkbox("Стохастичность", value=True)
    use_delay = st.sidebar.checkbox("Задержка", value=True)

elif model == "Модель Лесли":
    n = st.sidebar.number_input("Число возрастных классов", min_value=2, max_value=10, value=3)
    with st.sidebar.expander("Коэффициенты рождаемости"):
        fertility = [st.number_input(f"f_{i}", min_value=0.0, value=0.5) for i in range(n)]
    with st.sidebar.expander("Вероятности выживания"):
        survival = [st.number_input(f"s_{i}", min_value=0.0, max_value=1.0, value=0.8) for i in range(n-1)]
    with st.sidebar.expander("Начальная популяция"):
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

else:
    configs_count = st.sidebar.number_input("Количество конфигураций", min_value=1, max_value=5, value=1)
    config_params = []
    for i in range(configs_count):
        st.sidebar.markdown(f"**Конфигурация #{i+1}**")
        N0_i = st.sidebar.number_input(f"N0 #{i+1}", min_value=0.0, value=10.0)
        r_i = st.sidebar.number_input(f"r #{i+1}", min_value=0.0, value=0.1)
        K_i = st.sidebar.number_input(f"K #{i+1}", min_value=1.0, value=100.0)
        config_params.append((N0_i, r_i, K_i))

if st.sidebar.button("Симулировать"):
    with st.spinner("Симуляция..."):
        if model == "Логистический рост":
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

        elif model == "Гибридная модель":
            history = simulate_hybrid(
                N0_vec, T, fert_base, surv_base, K,
                r_fert, r_surv, delay_fert, delay_surv,
                migration_rates, env_effect, stoch_intensity
            )
            df = pd.DataFrame(history, columns=[f"Возраст {i}" for i in range(n)])
            st.subheader("Гибридная модель - Динамика по возрастным классам")
            st.line_chart(df)
            total_pop = df.sum(axis=1)
            st.subheader("Гибридная модель - Общая численность популяции")
            st.line_chart(pd.DataFrame(total_pop, columns=["Общая численность"]))
            params_str = (f"Возрастные классы: {n}, K={K}, r_fert={r_fert}, r_surv={r_surv}, "
                        f"env_effect={env_effect}, stoch_intensity={stoch_intensity}\n"
                        f"fert_base={fert_base}, surv_base={surv_base}\n"
                        f"delay_fert={delay_fert}, delay_surv={delay_surv}\n"
                        f"migration_rates={migration_rates}")
            export_csv(df, 'hybrid_model', 'Гибридная модель', params_str)

        elif model == "Альтернативная гибридная модель":
            params = {
                "r": r,
                "K": K,
                "m": m,
                "immigration": immigration,
                "delay": delay,
                "noise_std": noise_std,
                "N0": N0,
            }
            population = simulate_hybrid_model(
                params,
                T,
                use_density_dependence=use_density_dependence,
                use_migration=use_migration,
                use_noise=use_noise,
                use_delay=use_delay
            )
            df = pd.DataFrame(population, columns=["Популяция"])
            st.subheader("Альтернативная гибридная модель")
            st.line_chart(df)
            params_str = (f"r={r}, K={K}, m={m}, immigration={immigration}, delay={delay}, "
                        f"noise_std={noise_std}, N0={N0}\n"
                        f"Факторы: плотность={use_density_dependence}, миграция={use_migration}, "
                        f"шум={use_noise}, задержка={use_delay}")
            export_csv(df, 'alternative_hybrid_model', 'Альтернативная гибридная модель', params_str)

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
            export_csv(df, 'leslie_matrix','Модель Лесли',history)

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
                    sigma_descriptions.append(f"σ={sigma} (N0={common['N0']}, r={common['r']}, K={common['K']}")
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
