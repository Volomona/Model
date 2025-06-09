import sys
import io
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
# import g4f # Убедитесь, что g4f установлен: pip install g4f

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Модели ---
@st.cache_data
def simulate_logistic(N0: float, r: float, K: float, T: int) -> np.ndarray:
    Ns = np.zeros(T + 1)
    Ns[0] = N0
    for t in range(T):
        current_N = Ns[t]
        if K <= 1e-9:
            next_N = current_N + r * current_N
        else:
            next_N = current_N + r * current_N * (1 - current_N / K)
        if not np.isfinite(next_N):
            Ns[t+1:] = np.nan
            break
        Ns[t+1] = max(0, next_N)
    return Ns

@st.cache_data
def simulate_ricker(N0: float, r: float, K: float, T: int) -> np.ndarray:
    Ns = np.zeros(T + 1)
    Ns[0] = N0
    for t in range(T):
        current_N = Ns[t]
        if K <= 1e-9:
            next_N = current_N * np.exp(r)
        else:
            next_N = current_N * np.exp(r * (1 - current_N / K))
        if not np.isfinite(next_N):
            Ns[t+1:] = np.nan
            break
        Ns[t+1] = max(0, next_N)
    return Ns

@st.cache_data
def simulate_leslie(N0_vec: list, fertility: list, survival: list, T: int) -> np.ndarray:
    n = len(N0_vec)
    N_history = np.zeros((T + 1, n))
    N_history[0, :] = N0_vec
    L = np.zeros((n, n))
    L[0, :] = fertility
    if n > 1 and len(survival) == n - 1:
        for i in range(n - 1):
            L[i+1, i] = survival[i]
    elif n > 1 and len(survival) != n -1:
        logger.error(f"Leslie matrix survival rates mis-sized. Expected {n-1}, got {len(survival)}")
        N_history[1:,:] = np.nan
        return N_history
    for t in range(T):
        N_history[t+1, :] = L @ N_history[t, :]
        N_history[t+1, N_history[t+1,:] < 0] = 0
    return N_history

@st.cache_data
def simulate_delay(N0: float, r: float, K: float, T: int, tau: int) -> np.ndarray:
    if tau <= 0: tau = 1
    Ns_history = np.full(T + tau + 1, N0)
    for t_sim_step in range(T):
        current_idx = tau + t_sim_step + 1
        N_t_formula_idx = tau + t_sim_step
        N_t_minus_tau_idx = t_sim_step
        N_t_val = Ns_history[N_t_formula_idx]
        N_t_minus_tau_val = Ns_history[N_t_minus_tau_idx]
        if K <= 1e-9:
            next_N = N_t_val * np.exp(r)
        else:
            next_N = N_t_val * np.exp(r * (1 - N_t_minus_tau_val / K))
        if not np.isfinite(next_N):
            Ns_history[current_idx:] = np.nan
            break
        Ns_history[current_idx] = max(0, next_N)
    return Ns_history[tau : T + tau + 1]

@st.cache_data
def simulate_stochastic(_base_sim_func, N0: float, r: float, K: float, T: int, sigma: float, repeats: int) -> np.ndarray:
    all_runs = np.zeros((repeats, T + 1))
    for i in range(repeats):
        deterministic_traj = _base_sim_func(N0, r, K, T)
        noise = np.random.normal(0, sigma, size=T + 1)
        noise[0] = 0
        noisy_traj = deterministic_traj + noise
        noisy_traj = np.clip(noisy_traj, 0, None)
        all_runs[i, :] = noisy_traj
    return all_runs

def export_csv_and_conditionally_analyze_g4f(data_df, filename_base, model_type_str, simulation_params_str, data_for_gpt_str, auto_analyze_gpt=False):
    csv_data = data_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"Скачать {filename_base}.csv",
        data=csv_data,
        file_name=f"{filename_base}.csv",
        mime="text/csv",
        key=f"download_csv_{filename_base}"
    )

    gpt_button_label = f"Анализировать данные с GPT ({filename_base})"
    # Если auto_analyze_gpt=True, то анализ запускается без кнопки, иначе по кнопке
    if auto_analyze_gpt or st.button(gpt_button_label, key=f"gpt_analyze_button_{filename_base}"):
        try:
            import g4f
            full_prompt = (
                f"Вы - научный сотрудник, анализирующий данные моделирования популяционной динамики. "
                f"Проанализируйте следующие результаты симуляции.\n"
                f"Тип модели: {model_type_str}\n"
                f"Параметры симуляции: {simulation_params_str}\n"
                f"Данные траектории (или их сводка):\n{data_for_gpt_str}\n\n"
                f"Ваш анализ (формат Markdown, без запроса уточнений, как будто это единственный ответ):"
            )
            
            MAX_PROMPT_LENGTH = 12000 
            if len(full_prompt) > MAX_PROMPT_LENGTH:
                chars_to_keep_data = MAX_PROMPT_LENGTH - (len(full_prompt) - len(data_for_gpt_str))
                data_for_gpt_str_truncated = data_for_gpt_str[:max(0,chars_to_keep_data)] + "\n...(данные обрезаны)" if chars_to_keep_data > 100 else "(Данные слишком объемны)"
                full_prompt = (
                    f"Вы - научный сотрудник, анализирующий данные моделирования популяционной динамики. "
                    f"Проанализируйте следующие результаты симуляции.\n"
                    f"Тип модели: {model_type_str}\n"
                    f"Параметры симуляции: {simulation_params_str}\n"
                    f"Данные траектории (или их сводка):\n{data_for_gpt_str_truncated}\n\n"
                    f"Ваш анализ (формат Markdown, без запроса уточнений, как будто это единственный ответ):"
                )

            with st.spinner("GPT анализирует данные... Это может занять некоторое время."):
                response = g4f.ChatCompletion.create(
                    model=g4f.models.gpt_3_5_turbo,
                    messages=[{"role": "user", "content": full_prompt}],
                )
            
            container = st.container(border=True)
            container.subheader(f"Анализ от GPT для: {filename_base}")
            container.markdown(str(response))

        except ImportError:
            st.error("Библиотека g4f не найдена. Пожалуйста, установите ее: pip install g4f")
        except Exception as e:
            st.error(f"Ошибка при обращении к GPT: {e}")
            logger.error(f"GPT Error: {e}", exc_info=True)


st.set_page_config(page_title="Population Dynamics Simulator", layout="wide")
st.title("🌱 Симулятор Популяционной Динамики")

model_info = {
    "Логистический рост": "Классическая логистическая карта: $N_{t+1} = N_t + r N_t (1 - N_t/K)$.",
    "Модель Рикера": "Экспоненциальный рост с плотностной регуляцией: $N_{t+1} = N_t \exp(r(1 - N_t/K))$.",
    "Модель Лесли": "Возрастно-структурная модель: $N_{t+1} = L N_t$.",
    "Модель с задержкой": "Модель Рикера с запаздыванием: $N_{t+1} = N_t \exp(r(1 - N_{t-\tau}/K))$.",
    "Стохастическая симуляция": "Добавляет аддитивный гауссов шум к детерминированной модели.",
}
st.sidebar.info("Выберите модель и установите параметры ниже.")

model_choice = st.sidebar.selectbox("Выберите модель:", list(model_info.keys()), key="model_select")
st.sidebar.caption(model_info[model_choice])

st.sidebar.markdown("### Общие параметры")
T_sim_steps = st.sidebar.number_input("Шаги времени (T)", min_value=10, max_value=1000, value=100, key="T_sim")

default_N0, default_r, default_K = 10.0, 0.1, 100.0

params_common = {}
if model_choice != "Модель Лесли":
    params_common['N0'] = st.sidebar.number_input("Начальная популяция N0", min_value=0.0, value=default_N0, format="%.2f", key="N0_common")
    params_common['r'] = st.sidebar.number_input("Темп роста r", min_value=-2.0, max_value=4.0, value=default_r, format="%.2f", step=0.01, key="r_common")
    params_common['K'] = st.sidebar.number_input("Емкость K", min_value=0.0, value=default_K, format="%.2f", key="K_common")

if model_choice == "Модель с задержкой":
    delay_tau_values = st.sidebar.multiselect(
        "Значения задержки (τ)", options=list(range(1, 21)), default=[1, 2, 5], key="tau_multiselect"
    )
elif model_choice == "Модель Лесли":
    leslie_n_classes = st.sidebar.number_input("Число возрастных классов", min_value=1, max_value=15, value=3, key="leslie_n")
    with st.sidebar.expander("Коэффициенты рождаемости (f_i)"):
        leslie_fertility = [st.number_input(f"f_{i}", min_value=0.0, value=0.5 if i<2 else 0.2, format="%.2f", key=f"leslie_f_{i}") for i in range(leslie_n_classes)]
    leslie_survival = []
    if leslie_n_classes > 1 :
        with st.sidebar.expander("Вероятности выживания (s_i)"):
            leslie_survival = [st.number_input(f"s_{i} (из {i} в {i+1})", min_value=0.0, max_value=1.0, value=0.8, format="%.2f", key=f"leslie_s_{i}") for i in range(leslie_n_classes - 1)]
    with st.sidebar.expander("Начальная популяция по классам (N0_i)"):
        leslie_N0_vec = [st.number_input(f"N0_{i}", min_value=0.0, value=10.0, format="%.2f", key=f"leslie_N0_{i}") for i in range(leslie_n_classes)]
elif model_choice == "Стохастическая симуляция":
    stoch_repeats = st.sidebar.number_input("Число повторений", min_value=1, max_value=500, value=50, key="stoch_repeats")
    stoch_sigma_values = st.sidebar.multiselect(
        "Значения шума (σ, абсолютное стандартное отклонение)", options=[0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0], default=[0.1, 0.5], key="stoch_sigma_multiselect"
    )
    stoch_base_model_name = st.sidebar.selectbox("Основная детерминированная модель:", ["Логистический рост", "Модель Рикера"], key="stoch_base_model")
    stoch_base_sim_func = simulate_logistic if stoch_base_model_name == "Логистический рост" else simulate_ricker
else: 
    multi_configs_count = st.sidebar.number_input("Количество конфигураций для сравнения", min_value=1, max_value=5, value=1, key="multi_conf_count")
    multi_config_params_list = []
    shared_N0_for_multi = params_common.get('N0', default_N0)
    for i in range(multi_configs_count):
        expander_title = f"Конфигурация #{i+1}" if multi_configs_count > 1 else "Параметры симуляции"
        with st.sidebar.expander(expander_title, expanded=(multi_configs_count == 1)):
            r_i = st.number_input(f"r #{i+1}", min_value=-2.0, max_value=4.0, value=params_common.get('r', default_r) + i*0.2, format="%.2f", step=0.01, key=f"r_multi_{i}")
            K_i = st.number_input(f"K #{i+1}", min_value=0.0, value=params_common.get('K', default_K), format="%.2f", key=f"K_multi_{i}")
            multi_config_params_list.append({'N0': shared_N0_for_multi, 'r': r_i, 'K': K_i})

if st.sidebar.button("Запустить симуляцию", type="primary", key="run_simulation_button"):
    st.header(f"Результаты: {model_choice}")
    
    # Флаг для автоматического анализа GPT. Установите в True для мгновенного анализа.
    # Рекомендуется False для интерактивности.
    PERFORM_GPT_ANALYSIS_AUTOMATICALLY = False # ИЗМЕНИТЕ ЭТО, ЕСЛИ НУЖЕН АВТОМАТИЧЕСКИЙ АНАЛИЗ

    with st.spinner("Симуляция выполняется..."):
        # --- Логистический рост ---
        if model_choice == "Логистический рост":
            fig, ax = plt.subplots(figsize=(12, 7))
            all_trajs_dict = {}
            sim_details_parts = []
            for idx, cfg in enumerate(multi_config_params_list):
                traj = simulate_logistic(cfg['N0'], cfg['r'], cfg['K'], T_sim_steps)
                label = f"N0={cfg['N0']:.1f}, r={cfg['r']:.2f}, K={cfg['K']:.1f}"
                all_trajs_dict[label] = traj
                ax.plot(traj, label=label)
                sim_details_parts.append(f"Конф. {idx+1}: N0={cfg['N0']:.1f}, r={cfg['r']:.2f}, K={cfg['K']:.1f}")
            
            ax.set_xlabel("Время (t)")
            ax.set_ylabel("Численность популяции (N)")
            ax.set_title(f"Динамика популяции: {model_choice}")
            if any(ax.get_legend_handles_labels()): ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax.grid(True, linestyle=':', alpha=0.7)
            st.pyplot(fig)
            plt.close(fig)

            data_to_export_df = pd.DataFrame(all_trajs_dict)
            if not data_to_export_df.empty:
                simulation_details_for_gpt = "\n".join(sim_details_parts)
                data_as_string_for_gpt = data_to_export_df.to_string(max_rows=20, max_cols=7)
                export_csv_and_conditionally_analyze_g4f(data_to_export_df, "logistic_growth", model_choice, 
                                                         simulation_details_for_gpt, data_as_string_for_gpt,
                                                         auto_analyze_gpt=PERFORM_GPT_ANALYSIS_AUTOMATICALLY)

        # --- Модель Рикера ---
        elif model_choice == "Модель Рикера":
            fig, ax = plt.subplots(figsize=(12, 7))
            all_trajs_dict = {}
            sim_details_parts = []
            for idx, cfg in enumerate(multi_config_params_list):
                traj = simulate_ricker(cfg['N0'], cfg['r'], cfg['K'], T_sim_steps)
                label = f"N0={cfg['N0']:.1f}, r={cfg['r']:.2f}, K={cfg['K']:.1f}"
                all_trajs_dict[label] = traj
                ax.plot(traj, label=label)
                sim_details_parts.append(f"Конф. {idx+1}: N0={cfg['N0']:.1f}, r={cfg['r']:.2f}, K={cfg['K']:.1f}")

            ax.set_xlabel("Время (t)")
            ax.set_ylabel("Численность популяции (N)")
            ax.set_title(f"Динамика популяции: {model_choice}")
            if any(ax.get_legend_handles_labels()): ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax.grid(True, linestyle=':', alpha=0.7)
            st.pyplot(fig)
            plt.close(fig)

            data_to_export_df = pd.DataFrame(all_trajs_dict)
            if not data_to_export_df.empty:
                simulation_details_for_gpt = "\n".join(sim_details_parts)
                data_as_string_for_gpt = data_to_export_df.to_string(max_rows=20, max_cols=7)
                export_csv_and_conditionally_analyze_g4f(data_to_export_df, "ricker_model", model_choice,
                                                         simulation_details_for_gpt, data_as_string_for_gpt,
                                                         auto_analyze_gpt=PERFORM_GPT_ANALYSIS_AUTOMATICALLY)
        
        # --- Модель с задержкой ---
        elif model_choice == "Модель с задержкой":
            if not delay_tau_values:
                st.warning("Выберите хотя бы одно значение τ для модели с задержкой.")
            else:
                fig, ax = plt.subplots(figsize=(12, 7))
                all_trajs_dict = {}
                sim_details_parts = [f"Общие параметры: N0={params_common['N0']:.1f}, r={params_common['r']:.2f}, K={params_common['K']:.1f}"]
                for tau_i in delay_tau_values:
                    traj = simulate_delay(params_common['N0'], params_common['r'], params_common['K'], T_sim_steps, tau_i)
                    label = f"τ = {tau_i}"
                    all_trajs_dict[label] = traj
                    ax.plot(traj, label=label)
                    sim_details_parts.append(f"Траектория для τ={tau_i}")

                ax.set_xlabel("Время (t)")
                ax.set_ylabel("Численность популяции (N)")
                ax.set_title(f"Динамика популяции: {model_choice}")
                if any(ax.get_legend_handles_labels()): ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                ax.grid(True, linestyle=':', alpha=0.7)
                st.pyplot(fig)
                plt.close(fig)
                
                data_to_export_df = pd.DataFrame(all_trajs_dict)
                if not data_to_export_df.empty:
                    simulation_details_for_gpt = "\n".join(sim_details_parts)
                    data_as_string_for_gpt = data_to_export_df.to_string(max_rows=20, max_cols=7)
                    export_csv_and_conditionally_analyze_g4f(data_to_export_df, "delay_model", model_choice,
                                                             simulation_details_for_gpt, data_as_string_for_gpt,
                                                             auto_analyze_gpt=PERFORM_GPT_ANALYSIS_AUTOMATICALLY)
        
        # --- Модель Лесли ---
        elif model_choice == "Модель Лесли":
            fig, ax = plt.subplots(figsize=(12, 7))
            history_leslie = simulate_leslie(leslie_N0_vec, leslie_fertility, leslie_survival, T_sim_steps)
            df_columns = [f"Класс {i}" for i in range(leslie_n_classes)]
            data_to_export_df = pd.DataFrame(history_leslie, columns=df_columns)
            
            for col in data_to_export_df.columns:
                ax.plot(data_to_export_df[col], label=col)
            if leslie_n_classes > 1:
                 data_to_export_df['Суммарная'] = data_to_export_df.sum(axis=1)
                 ax.plot(data_to_export_df['Суммарная'], label="Суммарная", linestyle='--', color='black')

            ax.set_xlabel("Время (t)")
            ax.set_ylabel("Численность популяции (N)")
            ax.set_title(f"Динамика популяции: {model_choice}")
            if any(ax.get_legend_handles_labels()): ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax.grid(True, linestyle=':', alpha=0.7)
            st.pyplot(fig)
            plt.close(fig)

            L_matrix = np.zeros((leslie_n_classes, leslie_n_classes))
            L_matrix[0, :] = leslie_fertility
            if leslie_n_classes > 1 and len(leslie_survival) == leslie_n_classes -1:
                for i in range(leslie_n_classes - 1): L_matrix[i+1, i] = leslie_survival[i]
            
            lambda_max_str = "Не рассчитано"
            try:
                eigenvalues = np.linalg.eigvals(L_matrix)
                dominant_eigenvalue = np.max(np.abs(eigenvalues))
                lambda_max_str = f"{dominant_eigenvalue:.4f}"
                st.write(f"**Доминантное собственное число (λ_max):** {lambda_max_str}")
                if dominant_eigenvalue > 1 + 1e-6: st.success(f"Популяция растет (λ_max > 1).")
                elif dominant_eigenvalue < 1 - 1e-6: st.warning(f"Популяция вымирает (λ_max < 1).")
                else: st.info(f"Популяция близка к стабильной (λ_max ≈ 1).")
            except np.linalg.LinAlgError:
                st.error("Не удалось рассчитать собственные числа матрицы Лесли.")
            
            if not data_to_export_df.empty:
                simulation_details_for_gpt = (f"N0 по классам: {leslie_N0_vec}\n"
                                              f"Рождаемость f: {leslie_fertility}\n"
                                              f"Выживаемость s: {leslie_survival if leslie_n_classes > 1 else 'N/A'}\n"
                                              f"λ_max: {lambda_max_str}")
                data_as_string_for_gpt = data_to_export_df.to_string(max_rows=15, max_cols=5)
                export_csv_and_conditionally_analyze_g4f(data_to_export_df, "leslie_model", model_choice,
                                                         simulation_details_for_gpt, data_as_string_for_gpt,
                                                         auto_analyze_gpt=PERFORM_GPT_ANALYSIS_AUTOMATICALLY)

        # --- Стохастическая симуляция ---
        elif model_choice == "Стохастическая симуляция":
            if not stoch_sigma_values:
                st.warning("Выберите хотя бы одно значение σ для стохастической симуляции.")
            else:
                fig, ax = plt.subplots(figsize=(12, 7)) # Основной график для всех траекторий
                st.subheader(f"Траектории для {stoch_base_model_name} (N0={params_common['N0']:.1f}, r={params_common['r']:.2f}, K={params_common['K']:.1f})")
                all_means_dict = {}
                sim_details_parts = [f"Базовая модель: {stoch_base_model_name}",
                                     f"Общие параметры: N0={params_common['N0']:.1f}, r={params_common['r']:.2f}, K={params_common['K']:.1f}",
                                     f"Число повторений на σ: {stoch_repeats}"]

                stoch_progress_bar = st.progress(0, text="Выполнение стохастических симуляций...")
                total_stoch_sim_count = len(stoch_sigma_values) * stoch_repeats
                sims_done_count = 0

                for i, sigma_val in enumerate(stoch_sigma_values):
                    current_sigma_runs = simulate_stochastic(
                        stoch_base_sim_func, params_common['N0'], params_common['r'], params_common['K'], 
                        T_sim_steps, sigma_val, stoch_repeats
                    )
                    sims_done_count += stoch_repeats
                    progress_percentage = sims_done_count / total_stoch_sim_count
                    stoch_progress_bar.progress(progress_percentage, text=f"Обработка σ={sigma_val:.3f} ({sims_done_count}/{total_stoch_sim_count} симуляций)")
                    
                    for run_idx in range(stoch_repeats):
                        ax.plot(current_sigma_runs[run_idx, :], color=f"C{i % 10}", alpha=max(0.02, 0.2/stoch_repeats), linewidth=0.7)
                    
                    mean_traj = np.mean(current_sigma_runs, axis=0)
                    label = f"Среднее (σ={sigma_val:.3f})"
                    all_means_dict[label] = mean_traj
                    ax.plot(mean_traj, color=f"C{i % 10}", linewidth=2.5, label=label)
                    sim_details_parts.append(f"Для σ={sigma_val:.3f}: показаны {stoch_repeats} траекторий и их среднее.")
                
                stoch_progress_bar.empty()

                ax.set_xlabel("Время (t)")
                ax.set_ylabel("Численность популяции (N)")
                ax.set_title(f"Стохастическая симуляция: {stoch_base_model_name}")
                if any(ax.get_legend_handles_labels()): ax.legend(loc='upper left', bbox_to_anchor=(1,1))
                ax.grid(True, linestyle=':', alpha=0.7)
                st.pyplot(fig) # Показываем график со всеми траекториями
                plt.close(fig)
                
                data_to_export_df = pd.DataFrame(all_means_dict)
                if not data_to_export_df.empty:
                    st.subheader("Средние траектории по уровням шума (σ):")
                    st.line_chart(data_to_export_df) # Отдельный график только для средних (как было у вас)
                    simulation_details_for_gpt = "\n".join(sim_details_parts)
                    data_as_string_for_gpt = data_to_export_df.to_string(max_rows=15, max_cols=7)
                    export_csv_and_conditionally_analyze_g4f(data_to_export_df, "stochastic_means", model_choice,
                                                             simulation_details_for_gpt, data_as_string_for_gpt,
                                                             auto_analyze_gpt=PERFORM_GPT_ANALYSIS_AUTOMATICALLY)
    # Конец блока with st.spinner
else:
    st.info("Добро пожаловать! Настройте параметры в боковой панели и нажмите 'Запустить симуляцию'.")

st.sidebar.markdown("---")
st.sidebar.info("Разработано Лией Ахметовой. ВКР.")
