import sys
import io
import logging
import time
from threading import Thread
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

#--------------------------
# 1. НАСТРОЙКА ЛОГИРОВАНИЯ
#--------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#--------------------------
# 2. КОНСТАНТЫ И КОНФИГУРАЦИИ
#--------------------------
HEATMAP_PARAM_CONFIGS = {
    "Логистический рост": {
        "params": ["r", "K", "N0"],
        "ranges": {
            "r": (0.0, 3.0),
            "K": (10.0, 1000.0),
            "N0": (1.0, 100.0)
        }
    },
    "Модель Рикера": {
        "params": ["r", "K", "N0"],
        "ranges": {
            "r": (0.0, 3.0),
            "K": (10.0, 1000.0),
            "N0": (1.0, 100.0)
        }
    },
    "Гибридная модель": {
        "params": ["r", "K", "env_effect", "stoch_intensity", "r_surv", "delay_fert", "delay_surv", "migration_rates"],
        "ranges": {
            "r": (0.0, 3.0),
            "K": (10.0, 1000.0),
            "env_effect": (-1.0, 1.0),
            "stoch_intensity": (0.0, 1.0),
            "r_surv": (0.0, 0.5),
            "delay_fert": (0, 5),
            "delay_surv": (0, 5),
            "migration_rates": (0.0, 1.0)
        },
        "multi_param": ["migration_rates", "delay_fert", "delay_surv"]
    },
    "Модель Лесли": {
        "params": ["fertility", "survival", "N0_vec"],
        "ranges": {
            "fertility": (0.0, 2.0),
            "survival": (0.0, 1.0),
            "N0_vec": (1.0, 100.0)
        }
    },
    "Модель с задержкой": {
        "params": ["r", "K", "tau"],
        "ranges": {
            "r": (0.0, 3.0),
            "K": (10.0, 1000.0),
            "tau": (1, 10)
        }
    },
    "Стохастическая симуляция": {
        "params": ["r", "K", "N0", "sigma", "repeats"],
        "ranges": {
            "r": (0.0, 3.0),
            "K": (10.0, 1000.0),
            "N0": (1.0, 100.0),
            "sigma": (0.0, 1.0),
            "repeats": (10, 200)
        }
    }
}

model_info = {
    "Гибридная модель": "Интегративная модель с возрастной структурой, плотностной зависимостью, задержками, стохастичностью и пространственной структурой.",
    "Логистический рост": "Классическая логистическая карта с предельной численностью K.",
    "Модель Рикера": "Экспоненциальный рост с зависимостью от плотности (Рикер).",
    "Модель Лесли": "Возрастная структура модели через матрицу Лесли.",
    "Модель с задержкой": "Популяция зависит от прошлого состояния (задержка τ).",
    "Стохастическая симуляция": "Добавляет гауссов шум к нескольким запускам.",
}

#--------------------------
# 3. МОДЕЛИ СИМУЛЯЦИИ
#--------------------------

def simulate_logistic(N0: float, r: float, K: float, T: int) -> np.ndarray:
    """Логистическая модель роста популяции"""
    Ns = [N0]
    for _ in range(T):
        Ns.append(Ns[-1] + r * Ns[-1] * (1 - Ns[-1] / K))
    return np.array(Ns)

def simulate_ricker(N0: float, r: float, K: float, T: int) -> np.ndarray:
    """Модель роста Рикера"""
    Ns = [N0]
    for _ in range(T):
        Ns.append(Ns[-1] * np.exp(r * (1 - Ns[-1] / K)))
    return np.array(Ns)

def simulate_leslie(N0_vec: list, fertility: list, survival: list, T: int) -> np.ndarray:
    """Модель Лесли с возрастной структурой"""
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
    """Модель с временной задержкой"""
    # Создаем историю с начальными значениями
    Ns = [N0] * (tau + 1)
    # Симулируем T шагов
    for t in range(tau, T + tau):
        N_next = Ns[t] * np.exp(r * (1 - Ns[t - tau] / K))
        Ns.append(N_next)
    return np.array(Ns[:T + 1])  # Возвращаем только T+1 точек

def simulate_stochastic(base_sim, *args, sigma: float = 0.1, repeats: int = 100) -> np.ndarray:
    """Стохастическая обертка для любой модели"""
    runs = []
    for i in range(repeats):
        traj = base_sim(*args)
        noise = np.random.normal(0, sigma, size=traj.shape)
        runs.append(np.clip(traj + noise, 0, None))
    return np.array(runs)

def simulate_hybrid(
    N0_vec: list, T: int,
    fert_base: list, surv_base: list,
    K: float, r: float, r_surv: float,
    delay_fert: list, delay_surv: list,
    migration_rates: list = None,
    env_effect: float = 0.0,
    stoch_intensity: float = 0.1,
    features: dict = None
) -> np.ndarray:
    """Гибридная модель с возрастной структурой и множеством параметров"""
    # Значения по умолчанию, если не переданы
    if features is None:
        features = {
            "Плотностная зависимость рождаемости": True,
            "Плотностная зависимость выживаемости": True,
            "Задержки рождаемости": True,
            "Задержки выживаемости": True,
            "Миграция между группами": True,
            "Случайные колебания": True,
            "Влияние среды": True
        }

    n = len(N0_vec)
    N = np.array(N0_vec, dtype=float)
    history = [N.copy()]

    # Буфер для истории (для задержек)
    buffer_size = (max(delay_fert) if features["Задержки рождаемости"] else 0) + \
                  (max(delay_surv) if features["Задержки выживаемости"] else 0) + 1
    buffer = [N.copy()] * buffer_size

    # Миграционная матрица (по умолчанию нет миграции)
    if migration_rates is None or not features["Миграция между группами"]:
        migration_rates = [0.0] * n

    total_pop = np.sum(N)

    for t in range(T):
        # Текущее состояние
        N_new = np.zeros(n)
        total_pop = sum(buffer[-1])

        # Стохастический компонент
        noise = (np.random.normal(0, stoch_intensity * np.sqrt(buffer[-1] + 1))
                 if features["Случайные колебания"] else np.zeros(n))

        # Влияние среды
        env_factor = (1.0 + env_effect * np.sin(t * 0.1)
                      if features["Влияние среды"] else 1.0)

        # Рождаемость с плотностной зависимостью и задержкой
        for i in range(n):
            # Определяем популяцию для расчета - с задержкой или текущая
            delayed_pop = (buffer[-delay_fert[i]][i]
                           if features["Задержки рождаемости"]
                           else buffer[-1][i])

            # Плотностная зависимость рождаемости
            density_effect = (np.exp(-r * (total_pop / K))
                              if features["Плотностная зависимость рождаемости"]
                              else 1.0)

            fertility = fert_base[i] * density_effect * env_factor
            N_new[0] += fertility * buffer[-1][i]

        # Выживаемость с плотностной зависимостью и задержкой
        for i in range(1, n):
            # Определяем популяцию для расчета - с задержкой или текущая
            delayed_pop = (buffer[-delay_surv[i - 1]][i - 1]
                           if features["Задержки выживаемости"]
                           else buffer[-1][i - 1])

            # Плотностная зависимость выживаемости
            density_effect = (np.exp(-r_surv * (delayed_pop / (K / n)))
                              if features["Плотностная зависимость выживаемости"]
                              else 1.0)

            survival = surv_base[i - 1] * density_effect * env_factor
            N_new[i] += survival * buffer[-1][i - 1]

        # Миграция между возрастными классами
        if features["Миграция между группами"]:
            migration = np.zeros(n)
            for i in range(n):
                outflow = buffer[-1][i] * migration_rates[i]
                migration[i] -= outflow
                # Распределяем оттоки равномерно по другим классам
                for j in range(n):
                    if i != j:
                        migration[j] += outflow / (n - 1)
            N_new += migration

        # Добавляем шум и ограничиваем минимальную популяцию
        N_new = np.clip(N_new + noise, 0, None)

        # Обновляем буфер и историю
        buffer.append(N_new)
        if len(buffer) > buffer_size:
            buffer.pop(0)

        history.append(N_new.copy())

    return np.array(history)

#--------------------------
# 4. ФУНКЦИИ АНАЛИЗА
#--------------------------

def analyze_dynamics(time_series, eps=0.01, window=20):
    """
    Комплексный анализ динамики временного ряда
    
    Args:
        time_series: временной ряд (numpy array)
        eps: порог для определения стационарности
        window: размер окна анализа
    """
    # Берем последние значения для анализа установившегося режима
    last_values = time_series[-window:]
    
    # Базовые статистики
    stats = {
        "Среднее значение": np.mean(last_values),
        "Стандартное отклонение": np.std(last_values),
        "Размах колебаний": np.max(last_values) - np.min(last_values),
        "Коэффициент вариации": np.std(last_values) / np.mean(last_values) if np.mean(last_values) != 0 else np.inf
    }
    
    # Анализ тренда
    trend = np.polyfit(range(len(last_values)), last_values, 1)[0]
    stats["Наличие тренда"] = "Растущий" if trend > eps else "Убывающий" if trend < -eps else "Отсутствует"
    
    # Анализ автокорреляции
    from scipy import signal
    acf = signal.correlate(last_values - np.mean(last_values), 
                         last_values - np.mean(last_values), mode='full') / len(last_values)
    acf = acf[len(acf)//2:]
    
    # Поиск периодичности
    peaks, properties = signal.find_peaks(acf, height=0.1)
    if len(peaks) > 1:
        stats["Период колебаний"] = int(np.mean(np.diff(peaks)))
        stats["Регулярность колебаний"] = 1 - np.std(np.diff(peaks)) / np.mean(np.diff(peaks))
    
    # Анализ предсказуемости
    from scipy.stats import entropy
    hist, _ = np.histogram(last_values, bins='auto', density=True)
    stats["Энтропия"] = entropy(hist)
    
    # Оценка показателя Ляпунова
    lyap = 0
    for i in range(len(last_values)-1):
        if abs(last_values[i]) > 1e-10:
            lyap += np.log(abs(last_values[i+1]/last_values[i]))
    stats["Показатель Ляпунова"] = lyap / (len(last_values)-1)
    
    # Определение режима
    if stats["Коэффициент вариации"] < eps:
        regime = "Стационарный режим"
        description = f"""
        Популяция находится в устойчивом состоянии.
        Среднее значение: {stats['Среднее значение']:.2f}
        Отклонения: менее {eps*100:.1f}% от среднего
        Прогноз: система останется стабильной
        """
    elif len(peaks) > 1 and stats["Регулярность колебаний"] > 0.9:
        regime = "Периодические колебания"
        description = f"""
        Популяция совершает регулярные колебания.
        Период: {stats['Период колебаний']} шагов
        Амплитуда: {stats['Размах колебаний']:.2f}
        Регулярность: {stats['Регулярность колебаний']:.2f}
        """
    elif stats["Показатель Ляпунова"] > 0 and stats["Энтропия"] > np.log(2):
        regime = "Хаотический режим"
        description = f"""
        Популяция демонстрирует хаотическое поведение.
        Показатель Ляпунова: {stats['Показатель Ляпунова']:.3f}
        Энтропия: {stats['Энтропия']:.2f}
        Предсказуемость: низкая
        """
    else:
        regime = "Нерегулярные колебания"
        description = f"""
        Популяция совершает нерегулярные колебания.
        Коэффициент вариации: {stats['Коэффициент вариации']:.2f}
        Тренд: {stats['Наличие тренда']}
        Предсказуемость: средняя
        """
    
    return regime, stats, description

def calculate_extinction_probability(trajectories, threshold=1.0, last_n_steps=10):
    """
    Рассчитывает вероятность вымирания по результатам симуляции.
    
    Args:
        trajectories: np.array - массив траекторий (может быть 1D или 2D)
        threshold: float - порог ниже которого считаем вымиранием
        last_n_steps: int - сколько последних шагов учитывать
        
    Returns:
        float: вероятность вымирания (0-1)
    """
    if len(trajectories.shape) == 1:
        trajectories = trajectories.reshape(1, -1)
    
    extinct = 0
    for traj in trajectories:
        if np.any(traj[-last_n_steps:] < threshold):
            extinct += 1
    
    return extinct / len(trajectories)

def analyze_multiple_trajectories(trajectories, model_name, config_params=None):
    """
    Анализ нескольких траекторий с разными конфигурациями
    
    Args:
        trajectories: словарь {название конфигурации: траектория}
        model_name: название модели
        config_params: параметры конфигураций
    """
    st.subheader(f"Анализ динамики {model_name}")
    
    for idx, (config_name, traj) in enumerate(trajectories.items()):
        with st.expander(f"Анализ конфигурации {idx + 1}", expanded=(idx == 0)):
            # Получаем параметры конфигурации
            if config_params:
                N0_i, r_i, K_i = config_params[idx]
                st.markdown(f"""
                **Параметры конфигурации:**
                - N₀ = {N0_i}
                - r = {r_i}
                - K = {K_i}
                """)
            import random
            # Анализируем динамику
            fig, regime, stats, description = visualize_dynamics(traj, f"{config_name}")
            st.plotly_chart(fig,key=f"{random.randint(111,11111111)}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Режим: {regime}**")
                st.markdown(description)
            
            with col2:
                st.markdown("**Количественные характеристики:**")
                for key, value in stats.items():
                    if isinstance(value, float):
                        st.markdown(f"- {key}: {value:.3f}")
                    else:
                        st.markdown(f"- {key}: {value}")

def send_to_gpt(typem, str):
    """Отправка данных на анализ GPT-модели"""
    import g4f
    response = g4f.ChatCompletion.create(
        model=g4f.models.gpt_4,
        messages=[{"role": "user", "content": f"Воспринимай график как данные точек.Проанализируй график или возможно несколько графиков популяционной модели.Ничего не проси уточнить. Это не чат ты пишешь 1 раз и всё.Обязательно форматируй текст по MakrDown. будто ты научный сотрудник. Тип модели:{typem} вот результат симуляции: {str}"}],
    )
    return response

#--------------------------
# 5. ФУНКЦИИ ВИЗУАЛИЗАЦИИ
#--------------------------

def visualize_dynamics(time_series, title=""):
    """Визуализация временного ряда с анализом"""
    regime, stats, description = analyze_dynamics(time_series)
    
    # График временного ряда
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=time_series,
        mode='lines',
        name='Динамика популяции'
    ))
    
    fig.update_layout(
        title=f"{title}<br><sup>{regime}</sup>",
        xaxis_title="Время",
        yaxis_title="Численность",
        height=500
    )
    
    return fig, regime, stats, description

def generate_heatmap(
    model_func, param_name, param_range, param_steps, time_steps, fixed_params, model_type="standard"
):
    """Создание тепловой карты зависимости динамики от параметра"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Генерируем значения параметра
    if param_name == "tau":
        param_values = np.linspace(int(param_range[0]), int(param_range[1]), param_steps, dtype=int)
    else:
        param_values = np.linspace(param_range[0], param_range[1], param_steps)
    
    # Массив для хранения результатов
    results = np.zeros((param_steps, time_steps))
    
    # Получаем размерность для модели Лесли
    if model_type == "leslie":
        # Используем глобальную переменную n
        leslie_n = n  # n определена глобально в основном коде
    elif model_type == "hybrid":
        leslie_n = len(fixed_params.get('N0_vec', []))
    
    # Симулируем для каждого значения параметра
    for i, param_val in enumerate(param_values):
        current_params = fixed_params.copy()
        
        # Специальная обработка параметров
        if model_type == "leslie":
            if param_name == "fertility":
                current_params["fertility"] = [float(param_val)] * leslie_n
            elif param_name == "survival":
                current_params["survival"] = [float(param_val)] * (leslie_n - 1)
            elif param_name == "N0_vec":
                current_params["N0_vec"] = [float(param_val)] * leslie_n
        elif model_type == "hybrid":
            if param_name == "migration_rates":
                current_params["migration_rates"] = [float(param_val)] * leslie_n
            elif param_name == "delay_fert":
                current_params["delay_fert"] = [int(param_val)] * leslie_n
            elif param_name == "delay_surv":
                current_params["delay_surv"] = [int(param_val)] * (leslie_n - 1)
            else:
                current_params[param_name] = float(param_val)
        elif param_name == "tau":
            current_params[param_name] = int(param_val)
        else:
            current_params[param_name] = float(param_val)
            
        try:
            # Запускаем симуляцию
            trajectory = model_func(**current_params)
            
            # Обработка результатов в зависимости от типа модели
            if model_type == "standard":
                results[i, :] = trajectory[-time_steps:]
            elif model_type in ["leslie", "hybrid"]:
                if len(trajectory.shape) > 1:
                    results[i, :] = np.sum(trajectory[-time_steps:], axis=1)
                else:
                    results[i, :] = trajectory[-time_steps:]
            
            # Обновляем прогресс
            progress = (i + 1) / param_steps
            progress_bar.progress(progress)
            status_text.text(f"Прогресс: {progress:.1%} ({param_name} = {param_val:.2f})")
            
        except Exception as e:
            st.error(f"Ошибка при параметре {param_name} = {param_val}: {str(e)}")
            st.write("Текущие параметры:", current_params)
            continue
    
    # Создаем тепловую карту
    fig = go.Figure(data=go.Heatmap(
        z=results,
        x=np.arange(time_steps),
        y=param_values,
        colorscale='Viridis',
        colorbar=dict(
            title=dict(
                text='Численность популяции',
                side='right'
            )
        )
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Тепловая карта зависимости от параметра {param_name}",
            x=0.5,
            y=0.95
        ),
        xaxis_title="Временные шаги",
        yaxis_title=param_name,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        width=800,
        height=600
    )
    
    fig.update_traces(
        hoverongaps=False,
        hovertemplate=(
            f"Время: %{{x}}<br>"
            f"{param_name}: %{{y:.2f}}<br>"
            f"Численность: %{{z:.2f}}<br>"
            "<extra></extra>"
        )
    )
    
    return fig, results

def generate_stochastic_heatmap(
    base_sim,
    param_name,
    param_range,
    param_steps,
    time_steps,
    fixed_params,
):
    """Специальная версия generate_heatmap для стохастической модели"""
    status_text = st.empty()
    
    param_values = np.linspace(param_range[0], param_range[1], param_steps)
    results = np.zeros((param_steps, time_steps))
    
    # Базовые параметры для детерминистической модели
    N0 = fixed_params.get("N0", 10.0)
    r = fixed_params.get("r", 0.1)
    K = fixed_params.get("K", 100.0)
    T = fixed_params.get("T", 100)
    sigma = fixed_params.get("sigma", 0.1)
    repeats = fixed_params.get("repeats", 50)
    
    for i, param_val in enumerate(param_values):
            # Определяем, какой параметр меняем
            if param_name == "sigma":
                results_array = simulate_stochastic(
                    base_sim, N0, r, K, T,
                    sigma=param_val,
                    repeats=repeats
                )
            elif param_name == "r":
                results_array = simulate_stochastic(
                    base_sim, N0, param_val, K, T,
                    sigma=sigma,
                    repeats=repeats
                )
            elif param_name == "K":
                results_array = simulate_stochastic(
                    base_sim, N0, r, param_val, T,
                    sigma=sigma,
                    repeats=repeats
                )
            elif param_name == "N0":
                results_array = simulate_stochastic(
                    base_sim, param_val, r, K, T,
                    sigma=sigma,
                    repeats=repeats
                )
            
            # Берем среднюю траекторию
            mean_trajectory = np.mean(results_array, axis=0)
            results[i, :] = mean_trajectory[-time_steps:]
    
    # Создаем тепловую карту
    fig = go.Figure(data=go.Heatmap(
        z=results,
        x=np.arange(time_steps),
        y=param_values,
        colorscale='Viridis',
        colorbar=dict(
            title=dict(
                text='Средняя численность популяции',
                side='right'
            )
        )
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Тепловая карта зависимости от параметра {param_name}",
            x=0.5,
            y=0.95
        ),
        xaxis_title="Временные шаги",
        yaxis_title=param_name,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        width=800,
        height=600
    )
    
    return fig, results

def plot_phase_portrait(signal, delay=1):
    """Построение фазового портрета временного ряда"""
    x = signal[delay:]
    y = signal[:-delay]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=3)))
    fig.update_layout(title="Фазовый портрет", xaxis_title="N(t)", yaxis_title=f"N(t - {delay})")
    st.plotly_chart(fig)

def plot_3d_surface_interactive(results_array, param_values, future_steps, param_to_vary):
    """Интерактивная 3D визуализация результатов симуляции"""
    X = np.arange(future_steps)
    Y = param_values
    X, Y = np.meshgrid(X, Y)
    Z = results_array

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig.update_layout(
        scene = dict(
            xaxis_title='Шаги времени',
            yaxis_title=param_to_vary.split('(')[0].strip(),
            zaxis_title='Общая численность'
        ),
        title=f"Интерактивный 3D-график: {param_to_vary.split('(')[0].strip()} / время / численность",
        autosize=True,
        margin=dict(l=40, r=40, b=40, t=40)
    )
    st.plotly_chart(fig, use_container_width=True)

def bifurcation_diagram_hybrid(param_name, param_range, steps, T_sim=100, current_params=None):
    """Построение бифуркационной диаграммы для гибридной модели"""
    param_values = np.linspace(param_range[0], param_range[1], steps)
    results = []
    
    # Получаем текущие параметры гибридной модели
    if current_params is None:
        current_params = {
            "N0_vec": N0_vec,
            "T": T_sim,
            "fert_base": fert_base,
            "surv_base": surv_base,
            "K": K,
            "r": r,
            "r_surv": r_surv,
            "delay_fert": delay_fert,
            "delay_surv": delay_surv,
            "migration_rates": migration_rates,
            "env_effect": env_effect,
            "stoch_intensity": stoch_intensity,
            "features": model_features
        }
    
    for val in param_values:
        sim_params = current_params.copy()
        
        # Изменяем нужный параметр
        if param_name in ["r", "r_surv", "K", "env_effect", "stoch_intensity"]:
            sim_params[param_name] = val
        elif param_name == "migration_rates":
            sim_params[param_name] = [val] * len(N0_vec)
        elif param_name == "delay_fert":
            sim_params[param_name] = [int(val)] * len(N0_vec)
        elif param_name == "delay_surv":
            sim_params[param_name] = [int(val)] * (len(N0_vec) - 1)
        
        # Запускаем симуляцию
        trajectory = simulate_hybrid(**sim_params)
        
        # Суммируем по всем возрастным группам для каждого шага времени
        total_pop = trajectory[-20:].sum(axis=1)  # последние 20 точек, сумма по всем группам
        
        for x in total_pop:
            results.append((val, x))
    
    df = pd.DataFrame(results, columns=[param_name, 'N'])
    fig = px.scatter(df, x=param_name, y='N', title=f"Бифуркационная диаграмма ({param_name})", opacity=0.3)
    st.plotly_chart(fig)

def export_csv(data, filename, typem, str):
    """Экспорт данных в CSV с аналитикой"""
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
    with st.expander(label="Анализ модели", expanded=True):
        container = st.container(border=True)
        container.write('Анализ модели:')
        container.write(send_to_gpt(typem, str))

#--------------------------
# 6. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
#--------------------------

def collect_current_params(model: str, param_to_vary: str) -> dict:
    """Собирает текущие параметры модели из сайдбара, исключая изменяемый параметр"""
    
    if model == "Логистический рост" or model == "Модель Рикера":
        params = {
            "N0": config_params[0][0],
            "r": config_params[0][1],
            "K": config_params[0][2],
            "T": T
        }
        if param_to_vary in params:
            del params[param_to_vary]
        return params
    
    elif model == "Гибридная модель":
        params = {
            "N0_vec": N0_vec,
            "T": T,
            "fert_base": fert_base,
            "surv_base": surv_base,
            "K": K,
            "r": r,
            "r_surv": r_surv,
            "delay_fert": delay_fert,
            "delay_surv": delay_surv,
            "migration_rates": migration_rates,
            "env_effect": env_effect,
            "stoch_intensity": stoch_intensity,
            "features": model_features
        }
        if param_to_vary in params:
            del params[param_to_vary]
        return params
    
    elif model == "Модель Лесли":
        params = {
            "N0_vec": N0_vec,
            "fertility": fertility,
            "survival": survival,
            "T": T
        }
        # Просто удаляем параметр целиком
        if param_to_vary in params:
            del params[param_to_vary]
        return params
    
    elif model == "Модель с задержкой":
        params = {
            "N0": common['N0'],
            "r": common['r'],
            "K": common['K'],
            "T": T,
            "tau": tau_values[0]
        }
        if param_to_vary in params:
            del params[param_to_vary]
        return params
    
    elif model == "Стохастическая симуляция":
        params = {
            "N0": common['N0'],
            "r": common['r'],
            "K": common['K'],
            "T": T,
            "sigma": sigma_values[0],
            "repeats": repeats,
            "base_sim": base_sim  # Добавляем базовую модель
        }
        # Не удаляем параметр, который будем варьировать
        return params

#--------------------------
# 7. НАСТРОЙКА UI И ЗАПУСК
#--------------------------

# Настройка страницы
st.set_page_config(page_title="Population Dynamics Simulator", layout="wide")
st.title("🌱 Симулятор Популяционной Динамики")

# Информация в сайдбаре
st.sidebar.info("Выберите модель и установите параметры ниже.")

# Выбор модели
model = st.sidebar.selectbox("Выберите модель:", list(model_info.keys()))
st.sidebar.caption(model_info[model])

# Общие параметры
st.sidebar.markdown("### Общие параметры")
T = st.sidebar.number_input("Шаги времени (T)", min_value=1, max_value=500, value=100)

# Инициализация общих параметров
common = {}
if model != "Модель Лесли":
    common['N0'] = st.sidebar.number_input("Начальная популяция N0", min_value=0.0, value=10.0)
    common['r'] = st.sidebar.number_input("Темп роста r", min_value=0.0, value=0.1)
    common['K'] = st.sidebar.number_input("Емкость K", min_value=1.0, value=100.0)

# Специфические параметры моделей
if model == "Модель с задержкой":
    tau_values = st.sidebar.multiselect(
        "Значения задержки (τ)",
        options=list(range(1, 11)),
        default=[1, 2]
    )
elif model == "Гибридная модель":
    n = st.sidebar.number_input(
        "Число возрастных групп",
        min_value=1,
        max_value=10,
        value=3,
        help="Например: 3 группы = молодые/взрослые/старые"
    )
    st.sidebar.markdown("### Параметры модели")

    model_features = {
            "Плотностная зависимость рождаемости": True,
            "Плотностная зависимость выживаемости": True,
            "Задержки рождаемости": True,
            "Задержки выживаемости": True,
            "Миграция между группами": True,
            "Случайные колебания": True,
            "Влияние среды": True
    }

    for feature, default in model_features.items():
        model_features[feature] = st.sidebar.toggle(feature, value=default)
    with st.sidebar.expander("Начальная численность"):
        st.markdown("""
            <div style="color: #666; font-size:0.9rem; margin-bottom:10px;">
            Сколько особей в каждой группе в начале моделирования
            </div>
            """, unsafe_allow_html=True)
        N0_vec = [
            st.number_input(
                f"🔢 Группа {i + 1} (молодые)" if i == 0 else
                f"🔢 Группа {i + 1} (взрослые)" if i == 1 else
                f"🔢 Группа {i + 1} (старые)",
                min_value=0.0,
                value=10.0
            ) for i in range(n)
        ]

    if model_features["Плотностная зависимость рождаемости"]:
        with st.sidebar.expander("Рождаемость"):
            st.markdown("""
                <div style="color: #666; font-size:0.9rem; margin-bottom:10px;">
                Сколько потомков производит одна особь из этой группы
                </div>
                """, unsafe_allow_html=True)
            fert_base = [
                st.number_input(
                    f"👶 Детей на 1 особь группы {i + 1}",
                    min_value=0.0,
                    value=0.5,
                    help=f"Например: 0.5 = 1 особь производит 0.5 потомков в среднем"
                ) for i in range(n)
            ]
    else:
        # Если параметр неактивен, можно либо не показывать ничего, либо задать дефолт
        fert_base = [0.0] * n  # или любое дефолтное безопасное значение

    if model_features["Плотностная зависимость выживаемости"]:
        with st.sidebar.expander("Выживаемость"):
            st.markdown("""
                <div style="color: #666; font-size:0.9rem; margin-bottom:10px;">
                Вероятность перехода в следующую возрастную группу
                </div>
                """, unsafe_allow_html=True)
            surv_base = [
                st.number_input(
                    f"🔄 Группа {i + 1} → Группа {i + 2}",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.8,
                    help=f"0.8 = 80% особей перейдут в следующую группу"
                ) for i in range(n - 1)
            ]
    else:
        surv_base = [0.0] * (n - 1)

    if model_features["Задержки рождаемости"]:
        with st.sidebar.expander("Задержка реакции рождаемости"):
            st.markdown("""
                <div style="color: #666; font-size:0.9rem; margin-bottom:10px;">
                Через сколько шагов рождаемость реагирует на изменения
                </div>
                """, unsafe_allow_html=True)
            delay_fert = [
                st.number_input(
                    f"⏳ Группа {i + 1} (шагов)",
                    min_value=0,
                    max_value=5,
                    value=1,
                    help="0 = мгновенная реакция, 1 = реагирует через 1 шаг"
                ) for i in range(n)
            ]
    else:
        delay_fert = [0] * n

    if model_features["Задержки выживаемости"]:
        with st.sidebar.expander("Задержка реакции выживаемости"):
            st.markdown("""
                <div style="color: #666; font-size:0.9rem; margin-bottom:10px;">
                Через сколько шагов выживаемость реагирует на изменения
                </div>
                """, unsafe_allow_html=True)
            delay_surv = [
                st.number_input(
                    f"⏳ Переход {i + 1}→{i + 2} (шагов)",
                    min_value=0,
                    max_value=5,
                    value=1
                ) for i in range(n - 1)
            ]
    else:
        delay_surv = [0] * (n - 1)

    if model_features["Миграция между группами"]:
        with st.sidebar.expander("Миграция между группами"):
            st.markdown("""
                <div style="color: #666; font-size:0.9rem; margin-bottom:10px;">
                Какая доля особей переходит в другие группы каждый шаг
                </div>
                """, unsafe_allow_html=True)
            migration_rates = [
                st.number_input(
                    f"🔄 Группа {i + 1} (доля мигрантов)",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.1,
                    help="0.1 = 10% особей уйдут в другие группы"
                ) for i in range(n)
            ]
    else:
        migration_rates = [0.0] * n

    st.sidebar.markdown("---")
    K = st.sidebar.number_input(
        "📊 Максимальная численность (K)",
        min_value=1.0,
        value=100.0,
        help="Предел, который среда может поддерживать"
    )
    r = st.sidebar.number_input(
        "📉 Влияние плотности на рождаемость",
        min_value=0.0,
        value=0.1,
        help="Чем больше, тем сильнее падает рождаемость при росте популяции"
    )
    r_surv = st.sidebar.number_input(
        "📉 Влияние плотности на выживаемость",
        min_value=0.0,
        value=0.05,
        help="Чем больше, тем сильнее падает выживаемость при росте популяции"
    )
    if model_features["Влияние среды"]:
        env_effect = st.sidebar.slider(
            "🌡️ Влияние среды",
            min_value=-1.0,
            max_value=1.0,
            value=0.2,
            help="-1: кризис, 0: нейтрально, +1: благоприятные условия"
        )
    else:
        env_effect = 0.0

    if model_features["Случайные колебания"]:
        stoch_intensity = st.sidebar.slider(
            "🎲 Случайные колебания",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            help="0: нет случайности, 1: сильные случайные изменения"
        )
    else:
        stoch_intensity = 0.0

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

else:
    configs_count = st.sidebar.number_input("Количество конфигураций", min_value=1, max_value=5, value=1)
    config_params = []
    for i in range(configs_count):
        st.sidebar.markdown(f"**Конфигурация #{i+1}**")
        N0_i = st.sidebar.number_input(f"N0 (начальная популяция) #{i+1}", min_value=0.0, value=10.0)
        r_i = st.sidebar.number_input(f"r (темп роста) #{i+1}", min_value=0.0, value=0.1)
        K_i = st.sidebar.number_input(f"K (емкость) #{i+1}", min_value=1.0, value=100.0)
        config_params.append((N0_i, r_i, K_i))

st.sidebar.markdown('---')

# Настройки тепловой карты
with st.sidebar.expander("Настройки тепловой карты", expanded=True):
    enable_heatmap = st.checkbox("Включить тепловую карту", value=True)
    
    if enable_heatmap:
        # Получаем конфигурацию для выбранной модели
        model_config = HEATMAP_PARAM_CONFIGS[model]
        
        # Выбор параметра для анализа
        param_to_vary = st.selectbox(
            "Параметр для анализа",
            options=model_config["params"],
            key="heatmap_param"
        )
        
        # Настройка диапазона значений
        st.markdown("### Диапазон значений параметра")
        
        if param_to_vary in model_config.get("multi_param", []):
                st.markdown(f"**Примечание**: Для параметра '{param_to_vary}' будет использовано одно значение для всех элементов")
                default_range = model_config["ranges"][param_to_vary]
                param_range = st.slider(
                    f"Диапазон {param_to_vary}",
                    min_value=float(default_range[0]),
                    max_value=float(default_range[1]),
                    value=(float(default_range[0]), float(default_range[1])),
                    step=0.1
                )
        else:
            # Для обычных параметров
            default_range = model_config["ranges"][param_to_vary]
            param_range = st.slider(
                f"Диапазон {param_to_vary}",
                min_value=float(default_range[0]),
                max_value=float(default_range[1]),
                value=(float(default_range[0]), float(default_range[1])),
                step=0.1
            )
        
        # Настройки разрешения тепловой карты
        if param_to_vary == "tau":
            param_range = st.slider(
                f"Диапазон {param_to_vary}",
                min_value=int(default_range[0]),
                max_value=int(default_range[1]),
                value=(int(default_range[0]), int(default_range[1])),
                step=1  # Шаг = 1 для целых чисел
            )
        
        st.markdown("### Настройки разрешения")
        param_steps = st.slider(
            "Количество шагов параметра",
            min_value=10,
            max_value=100,
            value=30,
            step=5
        )
        
        time_steps = st.slider(
            "Количество временных шагов",
            min_value=10,
            max_value=100,
            value=30,
            step=5
        )

# Кнопка симуляции и логика обработки
if st.sidebar.button("Симулировать"):
    with st.spinner("Симуляция..."):
        # ЛОГИСТИЧЕСКИЙ РОСТ
        if model == "Логистический рост":
            if configs_count == 1:
                # Одна конфигурация
                traj = simulate_logistic(config_params[0][0], config_params[0][1], config_params[0][2], T)
                df = pd.DataFrame(traj, columns=["Популяция"])
                st.subheader("Логистический рост")
                st.line_chart(df)
                
                # Тепловая карта (если включена)
                if enable_heatmap:
                    try:
                        current_params = collect_current_params(model, param_to_vary)
                        fig, results = generate_heatmap(
                            model_func=simulate_logistic,
                            param_name=param_to_vary,
                            param_range=param_range,
                            param_steps=param_steps,
                            time_steps=time_steps,
                            fixed_params=current_params,
                            model_type="standard"
                        )
                        st.plotly_chart(fig)
                        
                        # 3D визуализация
                        param_values = np.linspace(param_range[0], param_range[1], param_steps)
                        plot_3d_surface_interactive(results, param_values, time_steps, param_to_vary)
                    except Exception as e:
                        st.error(f"Ошибка при генерации тепловой карты: {str(e)}")
                
                # Дополнительная аналитика
                ext_prob = calculate_extinction_probability(traj.reshape(1, -1))
                st.metric("Вероятность вымирания", f"{ext_prob:.1%}")
                
                # Визуализация динамики
                fig, regime, stats, description = visualize_dynamics(traj, "Логистическая модель")
                st.plotly_chart(fig)
                with st.expander("Подробный анализ динамики"):
                    st.markdown(f"**Режим: {regime}**")
                    st.markdown(description)
                    st.markdown("**Количественные характеристики:**")
                    for key, value in stats.items():
                        if isinstance(value, float):
                            st.markdown(f"- {key}: {value:.3f}")
                        else:
                            st.markdown(f"- {key}: {value}")
                
                # Экспорт данных
                export_csv(df, 'logistic_growth', 'Логистический рост',
                           f"Одна траектория: N0={config_params[0][0]}, r={config_params[0][1]}, K={config_params[0][2]}\nДанные:\n{traj}")
            else:
                # Несколько конфигураций
                all_trajs = {}
                config_descriptions = []
                for idx, (N0_i, r_i, K_i) in enumerate(config_params):
                    traj = simulate_logistic(N0_i, r_i, K_i, T)
                    all_trajs[f"Конфигурация #{idx + 1} (r={r_i}, K={K_i})"] = traj
                    config_descriptions.append(f"Конфигурация #{idx + 1}: N0={N0_i}, r={r_i}, K={K_i}")
                df = pd.DataFrame(all_trajs)
                st.subheader("Логистический рост - Несколько конфигураций")
                st.line_chart(df)
                
                # Анализ всех конфигураций
                analyze_multiple_trajectories(all_trajs, "логистической модели", config_params)
                
                # Расчет вероятности вымирания для каждой группы
                for idx, (N0_i, r_i, K_i) in enumerate(config_params):
                    traj = simulate_logistic(N0_i, r_i, K_i, T)
                    ext_prob = calculate_extinction_probability(traj.reshape(1, -1))
                    st.metric(f"Вероятность вымирания {idx+1} группы", f"{ext_prob:.1%}")
                
                # Тепловая карта
                if enable_heatmap:
                    try:
                        current_params = collect_current_params(model, param_to_vary)
                        fig, results = generate_heatmap(
                            model_func=simulate_logistic,
                            param_name=param_to_vary,
                            param_range=param_range,
                            param_steps=param_steps,
                            time_steps=time_steps,
                            fixed_params=current_params,
                            model_type="standard"
                        )
                        st.plotly_chart(fig)
                    except Exception as e:
                        st.error(f"Ошибка при генерации тепловой карты: {str(e)}")
                
                # Экспорт данных
                export_csv(df, 'logistic_growth_multiple', 'Логистический рост',
                           f"Множественные траектории:\n{'\n'.join(config_descriptions)}\nДанные:\n{all_trajs}")
        
        # ГИБРИДНАЯ МОДЕЛЬ
        elif model == "Гибридная модель":
            # Проверка на случай одной возрастной группы с задержками выживаемости
            if n == 1 and model_features['Задержки выживаемости']:
                st.warning("Для одной возрастной группы не может быть задержек выживаемости!\nПараметр был отключён.")
                model_features['Задержки выживаемости'] = False
            
            # Запуск симуляции
            history = simulate_hybrid(
                N0_vec, T, fert_base, surv_base, K,
                r, r_surv, delay_fert, delay_surv,
                migration_rates, env_effect, stoch_intensity,
                features=model_features
            )

            # Визуализация по возрастным классам
            df = pd.DataFrame(history, columns=[f"Возраст {i}" for i in range(n)])
            with st.expander("Гибридная модель - Разные классы", expanded=True):
                st.subheader("Гибридная модель - Динамика по возрастным классам")
                st.line_chart(df)
            
            # Общая статистика
            with st.expander("Гибридная модель - Общая статистика", expanded=False):
                total_pop = df.sum(axis=1)
                st.subheader("Гибридная модель - Общая численность популяции")
                st.line_chart(pd.DataFrame(total_pop, columns=["Общая численность"]))

            # Фазовый портрет и бифуркационная диаграмма
            plot_phase_portrait(history[:, 0])  # первая возрастная группа
            bifurcation_diagram_hybrid('r', (0.1, 3.0), 100, T_sim=T)

            
            # Вероятность вымирания
            extinction_prob = np.mean([np.any(run < 1e-3) for run in history])
            st.write(f"Вероятность вымирания: {extinction_prob:.2%}")
            
            # Анализ траекторий
            trajectories = {f"Возрастная группа {i+1}": history[:, i] for i in range(n)}
            total_pop = history.sum(axis=1)
            trajectories["Общая численность"] = total_pop
            
            analyze_multiple_trajectories(
                trajectories,
                "гибридной модели",
                config_params=[(N0_vec[i], r if i == 0 else r_surv, K/n) for i in range(n)] + [(sum(N0_vec), r, K)]
            )
            
            # Тепловая карта
            if enable_heatmap:
                current_params = collect_current_params(model, param_to_vary)
                fig, results = generate_heatmap(
                    model_func=simulate_hybrid,
                    param_name=param_to_vary,
                    param_range=param_range,
                    param_steps=param_steps,
                    time_steps=time_steps,
                    fixed_params=current_params,
                    model_type="hybrid"
                )
                st.plotly_chart(fig)
                
                # 3D визуализация
                param_values = np.linspace(param_range[0], param_range[1], param_steps)
                plot_3d_surface_interactive(results, param_values, time_steps, param_to_vary)
            
            # Экспорт данных
            params_str = (f"Возрастные классы: {n}, K={K}, r={r}, r_surv={r_surv}, "
                        f"env_effect={env_effect}, stoch_intensity={stoch_intensity}\n"
                        f"fert_base={fert_base}, surv_base={surv_base}\n"
                        f"delay_fert={delay_fert}, delay_surv={delay_surv}\n"
                        f"migration_rates={migration_rates}")
            export_csv(df, 'hybrid_model', 'Гибридная модель', params_str)

        # МОДЕЛЬ РИКЕРА
        elif model == "Модель Рикера":
            if configs_count == 1:
                # Одна конфигурация
                traj = simulate_ricker(config_params[0][0], config_params[0][1], config_params[0][2], T)
                df = pd.DataFrame(traj, columns=["Популяция"])
                st.subheader("Модель Рикера")
                st.line_chart(df)
                
                # Вероятность вымирания
                ext_prob = calculate_extinction_probability(traj.reshape(1, -1))
                st.metric("Вероятность вымирания", f"{ext_prob:.1%}")
                
                # Тепловая карта
                if enable_heatmap:
                    try:
                        current_params = collect_current_params(model, param_to_vary)
                        fig, results = generate_heatmap(
                            model_func=simulate_ricker,
                            param_name=param_to_vary,
                            param_range=param_range,
                            param_steps=param_steps,
                            time_steps=time_steps,
                            fixed_params=current_params,
                            model_type="standard"
                        )                    
                        st.plotly_chart(fig)
                        
                        # 3D визуализация
                        param_values = np.linspace(param_range[0], param_range[1], param_steps)
                        plot_3d_surface_interactive(results, param_values, time_steps, param_to_vary)
                    except Exception as e:
                        st.error(f"Ошибка при генерации тепловой карты: {str(e)}")
                
                # Визуализация динамики
                fig, regime, stats, description = visualize_dynamics(traj, "Модель Рикера")
                st.plotly_chart(fig)
                with st.expander("Подробный анализ динамики"):
                    st.markdown(f"**Режим: {regime}**")
                    st.markdown(description)
                    st.markdown("**Количественные характеристики:**")
                    for key, value in stats.items():
                        if isinstance(value, float):
                            st.markdown(f"- {key}: {value:.3f}")
                        else:
                            st.markdown(f"- {key}: {value}")
                
                # Экспорт данных
                export_csv(df, 'ricker_model', 'Модель Рикера',
                        f"Одна траектория: N0={config_params[0][0]}, r={config_params[0][1]}, K={config_params[0][2]}\nДанные:\n{traj}")
            else:
                # Несколько конфигураций
                all_trajs = {}
                config_descriptions = []
                for idx, (N0_i, r_i, K_i) in enumerate(config_params):
                    traj = simulate_ricker(N0_i, r_i, K_i, T)
                    all_trajs[f"Конфигурация #{idx + 1} (r={r_i}, K={K_i})"] = traj
                    config_descriptions.append(f"Конфигурация #{idx + 1}: N0={N0_i}, r={r_i}, K={K_i}")
                    ext_prob = calculate_extinction_probability(traj.reshape(1, -1))
                    st.metric(f"Вероятность вымирания {idx+1} группы", f"{ext_prob:.1%}")
                
                df = pd.DataFrame(all_trajs)
                st.subheader("Модель Рикера - Несколько конфигураций")
                st.line_chart(df)
                
                # Анализ траекторий
                analyze_multiple_trajectories(all_trajs, "модели Рикера", config_params)
                
                # Экспорт данных
                export_csv(df, 'ricker_model_multiple', 'Модель Рикера',
                           f"Множественные траектории:\n{'\n'.join(config_descriptions)}\nДанные:\n{all_trajs}")

        # МОДЕЛЬ С ЗАДЕРЖКОЙ
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
                    ext_prob = calculate_extinction_probability(traj.reshape(1, -1))
                    st.metric(f"Вероятность вымирания при τ={tau_i}", f"{ext_prob:.1%}")
                
                df = pd.DataFrame(all_trajs)
                st.subheader("Модель с задержкой - Разные τ")
                st.line_chart(df)
                
                # Тепловая карта
                if enable_heatmap:
                    current_params = collect_current_params(model, param_to_vary)
                    fig, results = generate_heatmap(
                        model_func=simulate_delay,
                        param_name=param_to_vary,
                        param_range=param_range,
                        param_steps=param_steps,
                        time_steps=time_steps,
                        fixed_params=current_params,
                        model_type="standard"
                    )
                    st.plotly_chart(fig)
                    
                    # 3D визуализация
                    param_values = np.linspace(param_range[0], param_range[1], param_steps)
                    plot_3d_surface_interactive(results, param_values, time_steps, param_to_vary)
                
                # Визуализация динамики
                for tau_i, traj in all_trajs.items():
                    fig, regime, stats, description = visualize_dynamics(traj, f"Модель с задержкой {tau_i}")
                    st.plotly_chart(fig)
                    with st.expander(f"Подробный анализ динамики для {tau_i}"):
                        st.markdown(f"**Режим: {regime}**")
                        st.markdown(description)
                        st.markdown("**Количественные характеристики:**")
                        for key, value in stats.items():
                            if isinstance(value, float):
                                st.markdown(f"- {key}: {value:.3f}")
                            else:
                                st.markdown(f"- {key}: {value}")
                
                # Экспорт данных
                export_csv(df, 'delay_model_multiple_tau', 'Модель с задержкой',
                           f"Траектории с разными задержками:\n{'\n'.join(tau_descriptions)}\nДанные:\n{all_trajs}")

        # МОДЕЛЬ ЛЕСЛИ
        elif model == "Модель Лесли":
            history = simulate_leslie(N0_vec, fertility, survival, T)
            df = pd.DataFrame(history, columns=[f"Возраст {i}" for i in range(n)])
            st.subheader("Модель Лесли")
            st.line_chart(df)
            
            # Анализ матрицы Лесли
            L = np.zeros((n, n))
            L[0, :] = fertility
            for i in range(1, n):
                L[i, i - 1] = survival[i - 1]
            lambda_val = np.max(np.real(np.linalg.eigvals(L)))
            st.write(f"Доминирующее собственное значение λ = {lambda_val:.3f}")
            
            # Вероятность вымирания
            total_pop = history.sum(axis=1)
            ext_prob = calculate_extinction_probability(total_pop.reshape(1, -1))
            st.metric("Вероятность вымирания", f"{ext_prob:.1%}")
            
            # Анализ траекторий
            trajectories = {f"Возрастная группа {i+1}": history[:, i] for i in range(n)}
            total_pop = history.sum(axis=1)
            trajectories["Общая численность"] = total_pop
            
            # Создаем параметры для анализа
            config_params = []
            for i in range(n):
                group_params = {
                    "N0": N0_vec[i],
                    "fertility": fertility[i],
                    "survival": survival[i-1] if i > 0 else None
                }
                config_params.append(group_params)
            
            for idx, (config_name, traj) in enumerate(trajectories.items()):
                with st.expander(f"Анализ {config_name}", expanded=(idx == 0)):
                    if idx < len(config_params):
                        params = config_params[idx]
                        st.markdown(f"""
                        **Параметры группы:**
                        - Начальная численность: {params['N0']}
                        - Рождаемость: {params['fertility']}
                        - Выживаемость: {params['survival'] if params['survival'] is not None else 'N/A'}
                        """)
                    
                    # Анализируем динамику
                    fig, regime, stats, description = visualize_dynamics(traj, config_name)
                    st.plotly_chart(fig)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Режим: {regime}**")
                        st.markdown(description)
                    
                    with col2:
                        st.markdown("**Количественные характеристики:**")
                        for key, value in stats.items():
                            if isinstance(value, float):
                                st.markdown(f"- {key}: {value:.3f}")
                            else:
                                st.markdown(f"- {key}: {value}")
            
            # Тепловая карта
            if enable_heatmap:
                try:
                    current_params = collect_current_params(model, param_to_vary)
                    # Убедимся, что все необходимые параметры присутствуют
                    if "N0_vec" not in current_params:
                        current_params["N0_vec"] = N0_vec
                    if "fertility" not in current_params:
                        current_params["fertility"] = fertility
                    if "survival" not in current_params:
                        current_params["survival"] = survival
                    current_params["T"] = T
                    
                    fig, results = generate_heatmap(
                        model_func=simulate_leslie,
                        param_name=param_to_vary,
                        param_range=param_range,
                        param_steps=param_steps,
                        time_steps=time_steps,
                        fixed_params=current_params,
                        model_type="leslie"
                    )
                    st.plotly_chart(fig)
                    
                    # 3D визуализация
                    param_values = np.linspace(param_range[0], param_range[1], param_steps)
                    plot_3d_surface_interactive(results, param_values, time_steps, param_to_vary)
                    
                except Exception as e:
                    st.error(f"Ошибка при генерации тепловой карты: {str(e)}")
                    st.write("Текущие параметры:", current_params)
            
            # Экспорт данных
            export_csv(df, 'leslie_matrix', 'Модель Лесли', history)

        # СТОХАСТИЧЕСКАЯ СИМУЛЯЦИЯ
        elif model == "Стохастическая симуляция":
            import matplotlib.pyplot as plt
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
                    
                    # Визуализация отдельных траекторий
                    for i in range(repeats):
                        ax.plot(results[i], alpha=0.1, linewidth=0.8)
                    
                    # Средняя траектория
                    mean_traj = results.mean(axis=0)
                    ax.plot(mean_traj, linewidth=2, label=f'σ={sigma}')
                    all_means[f"σ={sigma}"] = mean_traj
                    sigma_descriptions.append(f"σ={sigma} (N0={common['N0']}, r={common['r']}, K={common['K']})")
                
                ax.set_title(f"Стохастическая симуляция ({repeats} траекторий на сигму)")
                ax.legend()
                st.pyplot(fig)
                
                # Анализ средних траекторий
                means_df = pd.DataFrame(all_means)
                st.subheader("Средние траектории для разных уровней шума")
                st.line_chart(means_df)
                
                # Анализ вероятности вымирания
                ext_prob = calculate_extinction_probability(results)
                st.metric("Вероятность вымирания (общая)", f"{ext_prob:.1%}")
                
                # Анализ динамики для каждого уровня шума
                analyze_multiple_trajectories(all_means, "стохастической модели")
                
                # Анализ разброса и устойчивости
                for sigma in sigma_values:
                    sigma_trajectories = np.array([all_means[k] for k in all_means.keys() if k.startswith(f"σ={sigma}")])
                    mean_traj = sigma_trajectories.mean(axis=0)
                    std_traj = sigma_trajectories.std(axis=0)
                    
                    st.subheader(f"Анализ для σ={sigma}")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Анализ средней траектории
                        fig, regime, stats, description = visualize_dynamics(mean_traj, f"Средняя траектория (σ={sigma})")
                        st.plotly_chart(fig)
                        
                    with col2:
                        st.markdown("**Статистики разброса:**")
                        st.markdown(f"- Среднее стандартное отклонение: {std_traj.mean():.3f}")
                        st.markdown(f"- Максимальное отклонение: {std_traj.max():.3f}")
                        st.markdown(f"- Коэффициент вариации: {(std_traj/mean_traj).mean():.3f}")
                    
                    # Вероятность вымирания
                    ext_prob = calculate_extinction_probability(sigma_trajectories)
                    st.metric(f"Вероятность вымирания (σ={sigma})", f"{ext_prob:.1%}")
                
                # Тепловая карта
                if enable_heatmap:
                    current_params = collect_current_params(model, param_to_vary)
                    fig, results = generate_stochastic_heatmap(
                        base_sim=base_sim,
                        param_name=param_to_vary,
                        param_range=param_range,
                        param_steps=param_steps,
                        time_steps=time_steps,
                        fixed_params=current_params
                    )
                    st.plotly_chart(fig)
                    
                    # 3D визуализация
                    param_values = np.linspace(param_range[0], param_range[1], param_steps)
                    plot_3d_surface_interactive(results, param_values, time_steps, param_to_vary)
                
                # Экспорт данных
                export_csv(means_df, 'stochastic_simulation_means', 'Стохастическая модель',
                           f"Стохастические траектории с параметрами:\n{'\n'.join(sigma_descriptions)}\n"
                           f"Средние значения:\n{all_means}\n"
                           f"Базовые параметры: N0={common['N0']}, r={common['r']}, K={common['K']}")

# Футер
st.sidebar.markdown("---")
st.sidebar.info("Разработано Лией Ахметовой")
