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
# 1.2. СЛОВАРЬ ПЕРЕВОДОВ
#--------------------------

PARAM_TRANSLATIONS = {
    "r": "Темп роста популяции",
    "K": "Предельная численность среды",
    "N0": "Начальная численность",
    "tau": "Временная задержка",
    "sigma": "Уровень случайных колебаний",
    "env_effect": "Влияние окружающей среды",
    "stoch_intensity": "Интенсивность случайных изменений",
    "r_surv": "Влияние плотности на выживаемость",
    "delay_fert": "Задержка рождаемости",
    "delay_surv": "Задержка выживаемости",
    "migration_rates": "Скорость миграции",
    "fertility": "Коэффициент рождаемости",
    "survival": "Вероятность выживания",
    "repeats": "Количество повторений"
}

# Обратный словарь для преобразования обратно в технические термины
REVERSE_TRANSLATIONS = {v: k for k, v in PARAM_TRANSLATIONS.items()}

#--------------------------
# 2. КОНСТАНТЫ И КОНФИГУРАЦИИ
#--------------------------
HEATMAP_PARAM_CONFIGS = {
    "Логистический рост": {
        "params": ["Темп роста популяции", "Предельная численность среды", "Начальная численность"],
        "ranges": {
            "Темп роста популяции": (0.0, 3.0),
            "Предельная численность среды": (10.0, 1000.0),
            "Начальная численность": (1.0, 100.0)
        }
    },
    "Модель Рикера": {
        "params": ["Темп роста популяции", "Предельная численность среды", "Начальная численность"],
        "ranges": {
            "Темп роста популяции": (0.0, 3.0),
            "Предельная численность среды": (10.0, 1000.0),
            "Начальная численность": (1.0, 100.0)
        }
    },
    "Гибридная модель": {
        "params": [
            "Темп роста популяции",
            "Предельная численность среды",
            "Влияние окружающей среды",
            "Интенсивность случайных изменений",
            "Влияние плотности на выживаемость",
            "Задержка рождаемости",
            "Задержка выживаемости",
            "Скорость миграции"
        ],
        "ranges": {
            "Темп роста популяции": (0.0, 3.0),
            "Предельная численность среды": (10.0, 1000.0),
            "Влияние окружающей среды": (-1.0, 1.0),
            "Интенсивность случайных изменений": (0.0, 1.0),
            "Влияние плотности на выживаемость": (0.0, 0.5),
            "Задержка рождаемости": (0, 5),
            "Задержка выживаемости": (0, 5),
            "Скорость миграции": (0.0, 1.0)
        },
        "multi_param": ["Скорость миграции", "Задержка рождаемости", "Задержка выживаемости"]
    },
    "Модель Лесли": {
        "params": ["Коэффициент рождаемости", "Вероятность выживания", "Начальная численность"],
        "ranges": {
            "Коэффициент рождаемости": (0.0, 2.0),
            "Вероятность выживания": (0.0, 1.0),
            "Начальная численность": (1.0, 100.0)
        }
    },
    "Модель с задержкой": {
        "params": ["Темп роста популяции", "Предельная численность среды", "Временная задержка"],
        "ranges": {
            "Темп роста популяции": (0.0, 3.0),
            "Предельная численность среды": (10.0, 1000.0),
            "Временная задержка": (1, 10)
        }
    },
    "Стохастическая симуляция": {
        "params": [
            "Темп роста популяции",
            "Предельная численность среды",
            "Начальная численность",
            "Уровень случайных колебаний",
            "Количество повторений"
        ],
        "ranges": {
            "Темп роста популяции": (0.0, 3.0),
            "Предельная численность среды": (10.0, 1000.0),
            "Начальная численность": (1.0, 100.0),
            "Уровень случайных колебаний": (0.0, 1.0),
            "Количество повторений": (10, 200)
        }
    }
}

model_info = {
    "Гибридная модель": """
    Комплексная модель, учитывающая:
    - Возрастную структуру популяции
    - Влияние плотности на рождаемость и выживаемость
    - Временные задержки в процессах
    - Случайные факторы среды
    - Пространственное распределение
    """,
    "Логистический рост": """
    Классическая модель роста популяции с ограничением максимальной численности.
    Учитывает:
    - Естественный прирост
    - Конкуренцию за ресурсы
    - Предельную емкость среды
    """,
    "Модель Рикера": """
    Модель с нелинейной зависимостью от плотности популяции.
    Особенности:
    - Экспоненциальный рост при малой численности
    - Сильное самоограничение при большой численности
    - Возможность колебаний
    """,
    "Модель Лесли": """
    Возрастно-структурированная модель популяции.
    Описывает:
    - Различные возрастные группы
    - Специфичную для возраста рождаемость
    - Вероятности выживания между группами
    """,
    "Модель с задержкой": """
    Модель с учетом временного запаздывания процессов.
    Включает:
    - Отложенное влияние численности
    - Временные задержки в регуляции
    - Возможность колебательных режимов
    """,
    "Стохастическая симуляция": """
    Модель с учетом случайных факторов.
    Характеристики:
    - Множественные реализации процесса
    - Учет случайных воздействий
    - Анализ вероятностных исходов
    """
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


def analyze_phase_portrait(signal, delay=1):
    """
    Анализ фазового портрета для определения типа динамики
    
    Args:
        signal: временной ряд
        delay: задержка для анализа
        
    Returns:
        dict: результаты анализа
    """
    results = {
        "type": "Неопределенный",
        "characteristics": {},
        "explanation": "",
        "attractor_type": "Неопределенный"
    }
    
    # Создаем фазовый портрет
    x = signal[delay:]
    y = signal[:-delay]
    
    # Минимальная длина для надежного анализа
    if len(x) < 30:
        results["explanation"] = "Временной ряд слишком короткий для надежного анализа фазового портрета."
        return results
    
    # 1. Рассчитываем метрики для классификации типа динамики
    
    # 1.1 Коэффициент корреляции между x и y
    correlation = np.corrcoef(x, y)[0, 1]
    results["characteristics"]["correlation"] = correlation
    
    # 1.2 Плотность точек в фазовом пространстве
    # Разбиваем диапазон на сетку и считаем точки в ячейках
    range_x = max(x) - min(x)
    range_y = max(y) - min(y)
    
    # Определяем размер сетки в зависимости от длины сигнала
    grid_size = min(20, len(x) // 5)
    if grid_size < 2:
        grid_size = 2
    
    # Создаем 2D гистограмму
    hist, xedges, yedges = np.histogram2d(x, y, bins=grid_size)
    
    # Доля непустых ячеек
    non_empty_cells = np.sum(hist > 0) / (grid_size * grid_size)
    results["characteristics"]["non_empty_cells_ratio"] = non_empty_cells
    
    # 1.3 Расстояние между последовательными точками
    distances = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    
    # Коэффициент вариации расстояний
    if mean_distance > 0:
        cv_distance = std_distance / mean_distance
    else:
        cv_distance = 0
    
    results["characteristics"]["mean_distance"] = mean_distance
    results["characteristics"]["distance_variation"] = cv_distance
    
    # Простой анализ типа динамики на основе метрик
    
    # Признаки стационарной точки
    if np.max(signal[-len(signal)//3:]) - np.min(signal[-len(signal)//3:]) < 0.01 * (np.max(signal) - np.min(signal)):
        results["type"] = "Стационарная точка"
        results["attractor_type"] = "Точечный аттрактор"
        results["explanation"] = """
        **Стационарная точка (точечный аттрактор):**
        
        Система стабилизировалась в одном состоянии. На фазовом портрете это выглядит как скопление точек
        в одном месте. Система находится в устойчивом равновесии и не демонстрирует колебаний.
        """
    
    # Признаки предельного цикла (периодические колебания)
    elif non_empty_cells < 0.4 and abs(correlation) > 0.7 and cv_distance < 0.5:
        # Дополнительно проверяем количество уникальных состояний
        unique_states = len(np.unique(np.round(signal[-len(signal)//3:], 1)))
        if unique_states < 10:
            results["type"] = "Периодические колебания"
            results["attractor_type"] = "Предельный цикл"
            results["explanation"] = f"""
            **Периодические колебания (предельный цикл):**
            
            Система совершает регулярные повторяющиеся колебания. На фазовом портрете это выглядит как 
            замкнутая кривая или несколько точек, которые система посещает последовательно.
            """
    
    # Признаки хаоса
    elif non_empty_cells > 0.6 and cv_distance > 0.8:
        results["type"] = "Хаотическая динамика"
        results["attractor_type"] = "Странный аттрактор"
        results["explanation"] = """
        **Хаотическая динамика (странный аттрактор):**
        
        Система демонстрирует апериодическое поведение с чувствительностью к начальным условиям.
        На фазовом портрете это выглядит как сложная структура без регулярного паттерна.
        """
    
    # Признаки квазипериодических колебаний
    elif 0.4 <= non_empty_cells <= 0.6 and 0.5 <= cv_distance <= 0.8:
        results["type"] = "Квазипериодические колебания"
        results["attractor_type"] = "Тор"
        results["explanation"] = """
        **Квазипериодические колебания (тор):**
        
        Система совершает колебания, которые никогда точно не повторяются, но ограничены определенной областью.
        На фазовом портрете это выглядит как заполненное кольцо или тор.
        """
    
    # Признаки затухающих колебаний
    elif len(x) > 50 and np.std(signal[-20:]) < 0.5 * np.std(signal):
        results["type"] = "Затухающие колебания"
        results["attractor_type"] = "Точечный аттрактор с переходным процессом"
        results["explanation"] = """
        **Затухающие колебания:**
        
        Система начинает с колебаний, которые постепенно затухают по мере приближения к равновесию.
        На фазовом портрете это выглядит как спираль, сходящаяся к точке.
        """
    
    # Признаки случайного процесса
    elif non_empty_cells > 0.7 and cv_distance > 1.0:
        results["type"] = "Стохастический процесс"
        results["attractor_type"] = "Стохастический (не является аттрактором)"
        results["explanation"] = """
        **Стохастический процесс:**
        
        Система демонстрирует случайное поведение без ясной структуры. На фазовом портрете это выглядит
        как облако точек без определенного паттерна.
        """
    
    # Все остальные случаи
    else:
        results["type"] = "Сложная динамика"
        results["attractor_type"] = "Неопределенный"
        results["explanation"] = """
        **Сложная динамика:**
        
        Система демонстрирует поведение, которое не удается однозначно классифицировать.
        Возможно, это сочетание нескольких типов динамики или переходной режим.
        """
    
    return results

def plot_phase_portrait(signal, delays=[1], title="", plot_3d=False):
    """
    Улучшенная функция построения фазовых портретов с анализом
    
    Args:
        signal: временной ряд или матрица временных рядов
        delays: список задержек для построения (по умолчанию [1])
        title: название для заголовка
        plot_3d: строить ли 3D фазовый портрет (для задержки=2)
    
    Returns:
        fig: объект графика
        analysis: результаты анализа
    """
    # Обработка входных данных и проверка на ошибки
    try:
        # Преобразуем в numpy array для универсальности
        if isinstance(signal, pd.Series):
            signal = signal.values
        elif isinstance(signal, list):
            signal = np.array(signal)
        
        # Обработка многомерных данных (например, из модели Лесли)
        if len(signal.shape) > 1 and signal.shape[1] > 1:
            # Суммируем по всем группам
            signal = np.sum(signal, axis=1)
        else:
            signal = np.array(signal).flatten()
        
        # Если сигнал короткий, предупреждаем
        if len(signal) < 50:
            st.warning(f"Короткий временной ряд ({len(signal)} точек) может не показать полную динамику системы")
        
        # Определяем максимальную задержку
        max_delay = max(delays)
        if max_delay >= len(signal) - 10:
            max_delay = len(signal) // 5
            delays = [min(d, max_delay) for d in delays]
            st.warning(f"Задержка слишком велика, уменьшена до {max_delay}")
        
        # Сбор данных для 3D-графика (если нужен)
        if plot_3d and len(signal) > max_delay + 2:
            fig_3d = go.Figure()
            
            # Создаем 3D фазовый портрет с задержкой=2
            fig_3d.add_trace(go.Scatter3d(
                x=signal[2:],
                y=signal[1:-1],
                z=signal[:-2],
                mode='markers',
                marker=dict(size=3, color=np.arange(len(signal)-2), colorscale='Viridis')
            ))
            
            fig_3d.update_layout(
                title=f"3D фазовый портрет для {title}" if title else "3D фазовый портрет",
                scene=dict(
                    xaxis_title="N(t)",
                    yaxis_title="N(t-1)",
                    zaxis_title="N(t-2)"
                ),
                width=700,
                height=700
            )
            import random
            st.plotly_chart(fig_3d,key=random.randint(3275,32753275))
        
        # Создаем фазовые портреты для каждой задержки
        fig = go.Figure()
        
        for delay in delays:
            x = signal[delay:]
            y = signal[:-delay]
            
            fig.add_trace(go.Scatter(
                x=x, 
                y=y, 
                mode='markers+lines', 
                marker=dict(
                    size=4,
                    color=np.arange(len(x)),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Время")
                ),
                name=f"Задержка = {delay}"
            ))
        
        # Добавляем линию y=x для ориентира
        x_range = [min(signal), max(signal)]
        fig.add_trace(go.Scatter(
            x=x_range,
            y=x_range,
            mode='lines',
            line=dict(color='rgba(255,0,0,0.5)', dash='dash'),
            name='y=x'
        ))
        
        fig.update_layout(
            title=f"Фазовый портрет для {title}" if title else "Фазовый портрет",
            xaxis_title="N(t)",
            yaxis_title=f"N(t-{delays[0]})" if len(delays) == 1 else "N(t-delay)",
            legend_title="Параметры",
            width=700,
            height=600
        )
        
        # Проводим анализ фазового портрета
        analysis = analyze_phase_portrait(signal, delays[0])
        
        return fig, analysis
    
    except Exception as e:
        st.error(f"Ошибка при построении фазового портрета: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        
        # Возвращаем пустые значения в случае ошибки
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Ошибка построения фазового портрета")
        empty_analysis = {"type": "Ошибка", "characteristics": {}, "explanation": f"Произошла ошибка: {str(e)}", "attractor_type": "Неопределен"}
        return empty_fig, empty_analysis

def plot_and_analyze_phase_portrait(time_series, title="", delays=[1, 2], include_3d=True):
    """
    Общая функция для построения и анализа фазовых портретов
    
    Args:
        time_series: временной ряд или массив временных рядов
        title: название для заголовка
        delays: список задержек для построения
        include_3d: строить ли 3D фазовый портрет
    """
    # Проверка на None или пустые данные
    if time_series is None or (isinstance(time_series, (list, np.ndarray)) and len(time_series) == 0):
        st.warning("Нет данных для построения фазового портрета")
        return
    
    # Преобразуем в numpy array для универсальности
    if isinstance(time_series, pd.Series):
        time_series = time_series.values
    elif isinstance(time_series, list):
        time_series = np.array(time_series)
    
    # Для многомерных данных (например, из модели Лесли)
    is_multidimensional = len(time_series.shape) > 1 and time_series.shape[1] > 1
    
    # Проверка на минимальную длину для построения фазового портрета
    min_length = max(delays) + 5
    if len(time_series) < min_length:
        st.warning(f"Временной ряд слишком короткий для построения фазового портрета (длина {len(time_series)}, нужно минимум {min_length})")
        return
    
    if is_multidimensional:
        # Анализируем как общую сумму, так и каждую группу по отдельности
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Общая сумма
            total_series = np.sum(time_series, axis=1)
            fig, analysis = plot_phase_portrait(total_series, delays, 
                                               f"{title} (общая численность)", 
                                               plot_3d=include_3d)
            st.plotly_chart(fig)
        
        with col2:
            st.markdown(f"## Анализ фазового портрета")
            st.markdown(f"**Тип динамики:** {analysis['type']}")
            st.markdown(f"**Тип аттрактора:** {analysis['attractor_type']}")
            st.markdown(analysis['explanation'])
            
            # Показываем ключевые характеристики (не в expander!)
            st.markdown("### Технические характеристики:")
            for key, value in analysis["characteristics"].items():
                st.write(f"**{key}:** {value:.4f}")
        
        # Отдельные группы - не используем вложенные expanders!
        st.markdown("### Фазовые портреты отдельных групп")
        for i in range(time_series.shape[1]):
            st.markdown(f"#### Группа {i+1}")
            fig_group, analysis_group = plot_phase_portrait(time_series[:, i], [delays[0]], 
                                                          f"{title} (группа {i+1})", 
                                                          plot_3d=False)
            st.plotly_chart(fig_group)
            st.markdown(f"**Тип динамики:** {analysis_group['type']}")
            st.markdown(f"**Тип аттрактора:** {analysis_group['attractor_type']}")
    else:
        # Одномерные данные
        col1, col2 = st.columns([2, 1])
        
        with col1:
            import random
            fig, analysis = plot_phase_portrait(time_series, delays, title, plot_3d=True)
            st.plotly_chart(fig,key=random.randint(22222,3333333))
        
        with col2:
            st.markdown(f"## Анализ фазового портрета")
            st.markdown(f"**Тип динамики:** {analysis['type']}")
            st.markdown(f"**Тип аттрактора:** {analysis['attractor_type']}")
            st.markdown(analysis['explanation'])
            
            # Показываем ключевые характеристики (не в expander!)
            st.markdown("### Технические характеристики:")
            for key, value in analysis["characteristics"].items():
                st.write(f"**{key}:** {value:.4f}")
    
    # Добавляем информацию о фазовых портретах без использования expander
    st.markdown("### Как интерпретировать фазовый портрет:")
    st.markdown("""
    Фазовый портрет — это график, показывающий зависимость между значениями временного ряда 
    в разные моменты времени (обычно N(t) от N(t-1)). Это мощный инструмент анализа динамических систем.
    
    #### Основные типы аттракторов:
    
    1. **Точечный аттрактор:** Стабильное равновесие, система стремится к одной точке
    2. **Предельный цикл:** Периодические колебания, система движется по замкнутой кривой
    3. **Тор (квазипериодичность):** Колебания с несколькими несоизмеримыми частотами
    4. **Странный аттрактор:** Детерминированный хаос, непредсказуемое но ограниченное поведение
    """)
    
    return analysis


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
    
    # Перевод технических терминов в статистиках
    translated_stats = {}
    for key, value in stats.items():
        if key in PARAM_TRANSLATIONS:
            translated_stats[PARAM_TRANSLATIONS[key]] = value
        else:
            translated_stats[key] = value
    
    fig.update_layout(
        title=f"{title}<br><sup>{regime}</sup>",
        xaxis_title="Время",
        yaxis_title="Численность популяции",
        height=500
    )
    
    return fig, regime, translated_stats, description
def generate_heatmap(
    model_func, param_name, param_range, param_steps, time_steps, fixed_params, model_type="standard"
):
    """Создание тепловой карты зависимости динамики от параметра"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Преобразуем переведенное имя параметра в техническое
    tech_param_name = REVERSE_TRANSLATIONS.get(param_name, param_name)
    
    # Генерируем значения параметра
    if tech_param_name == "tau":
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
            if tech_param_name == "fertility":
                current_params["fertility"] = [float(param_val)] * leslie_n
            elif tech_param_name == "survival":
                current_params["survival"] = [float(param_val)] * (leslie_n - 1)
            elif tech_param_name == "N0_vec":
                current_params["N0_vec"] = [float(param_val)] * leslie_n
        elif model_type == "hybrid":
            if tech_param_name == "migration_rates":
                current_params["migration_rates"] = [float(param_val)] * leslie_n
            elif tech_param_name == "delay_fert":
                current_params["delay_fert"] = [int(param_val)] * leslie_n
            elif tech_param_name == "delay_surv":
                current_params["delay_surv"] = [int(param_val)] * (leslie_n - 1)
            else:
                current_params[tech_param_name] = float(param_val)
        elif tech_param_name == "tau":
            current_params[tech_param_name] = int(param_val)
        else:
            current_params[tech_param_name] = float(param_val)
            
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

# 1. Исправляем функцию generate_stochastic_heatmap
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
    progress_bar = st.progress(0)
    
    # Преобразуем переведенное имя параметра в техническое
    tech_param_name = REVERSE_TRANSLATIONS.get(param_name, param_name)
    
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
        try:
            # Обновляем прогресс
            progress = (i + 1) / param_steps
            progress_bar.progress(progress)
            status_text.text(f"Прогресс: {progress:.1%} ({param_name} = {param_val:.2f})")
            
            # Определяем, какой параметр меняем и вызываем с позиционными аргументами
            if tech_param_name == "sigma":
                results_array = simulate_stochastic(
                    base_sim, N0, r, K, T,
                    sigma=param_val,
                    repeats=repeats
                )
            elif tech_param_name == "r":
                results_array = simulate_stochastic(
                    base_sim, N0, param_val, K, T,
                    sigma=sigma,
                    repeats=repeats
                )
            elif tech_param_name == "K":
                results_array = simulate_stochastic(
                    base_sim, N0, r, param_val, T,
                    sigma=sigma,
                    repeats=repeats
                )
            elif tech_param_name == "N0":
                results_array = simulate_stochastic(
                    base_sim, param_val, r, K, T,
                    sigma=sigma,
                    repeats=repeats
                )
            elif tech_param_name == "repeats":
                # Особый случай - меняем количество повторений
                repeats_val = max(1, int(param_val))  # Минимум 1 повторение
                results_array = simulate_stochastic(
                    base_sim, N0, r, K, T,
                    sigma=sigma,
                    repeats=repeats_val
                )
            else:
                # Неизвестный параметр - используем значения по умолчанию
                st.warning(f"Неизвестный параметр для стохастической модели: {param_name}")
                results_array = simulate_stochastic(
                    base_sim, N0, r, K, T,
                    sigma=sigma,
                    repeats=repeats
                )
            
            # Берем среднюю траекторию
            mean_trajectory = np.mean(results_array, axis=0)
            results[i, :] = mean_trajectory[-time_steps:]
            
        except Exception as e:
            st.error(f"Ошибка при параметре {param_name} = {param_val}: {str(e)}")
            # В случае ошибки заполняем строку нулями
            results[i, :] = np.zeros(time_steps)
    
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

def plot_3d_surface_interactive(results_array, param_values, future_steps, param_to_vary):
    """Интерактивная 3D визуализация результатов симуляции"""
    X = np.arange(future_steps)
    Y = param_values
    X, Y = np.meshgrid(X, Y)
    Z = results_array

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    
    # Используем переведенное название параметра
    param_name = PARAM_TRANSLATIONS.get(param_to_vary, param_to_vary)
    
    fig.update_layout(
        scene = dict(
            xaxis_title='Временные шаги',
            yaxis_title=param_name,
            zaxis_title='Численность популяции'
        ),
        title=f"Интерактивная 3D-визуализация: {param_name} / время / численность",
        autosize=True,
        margin=dict(l=40, r=40, b=40, t=40)
    )
    st.plotly_chart(fig, use_container_width=True)

#--------------------------
# 4.1. ФУНКЦИИ БИФУРКАЦИОННОГО АНАЛИЗА
#--------------------------

def bifurcation_diagram(model_func, param_name, param_range, steps, T_sim=100, discard=50, current_params=None):
    """Построение бифуркационной диаграммы для любой модели"""
    param_values = np.linspace(param_range[0], param_range[1], steps)
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Преобразуем переведенное имя параметра в техническое
    tech_param_name = REVERSE_TRANSLATIONS.get(param_name, param_name)
    
    for i, val in enumerate(param_values):
        # Обновляем прогресс
        progress = (i + 1) / steps
        progress_bar.progress(progress)
        status_text.text(f"Прогресс: {progress:.1%} ({param_name} = {val:.2f})")
        
        sim_params = current_params.copy()
        
        # Изменяем нужный параметр, используя техническое имя
        if tech_param_name == "tau":
            sim_params[tech_param_name] = int(val)
        elif tech_param_name == "migration_rates":
            sim_params[tech_param_name] = [float(val)] * len(sim_params.get("N0_vec", [10]))
        elif tech_param_name == "delay_fert":
            sim_params[tech_param_name] = [int(val)] * len(sim_params.get("N0_vec", [10]))
        elif tech_param_name == "delay_surv":
            sim_params[tech_param_name] = [int(val)] * (len(sim_params.get("N0_vec", [10])) - 1)
        elif tech_param_name == "fertility":
            sim_params[tech_param_name] = [float(val)] * len(sim_params.get("N0_vec", [10]))
        elif tech_param_name == "survival":
            sim_params[tech_param_name] = [float(val)] * (len(sim_params.get("N0_vec", [10])) - 1)
        else:
            sim_params[tech_param_name] = val
        
        # Запускаем симуляцию
        try:
            trajectory = model_func(**sim_params)
            
            # Если результат многомерный (например, для модели Лесли), суммируем по всем группам
            if len(trajectory.shape) > 1 and trajectory.shape[1] > 1:
                trajectory = np.sum(trajectory, axis=1)
            
            # Берем только установившийся режим
            steady_state = trajectory[discard:]
            
            # Добавляем точки в результат
            for x in steady_state:
                results.append((val, x))
        except Exception as e:
            st.error(f"Ошибка при параметре {param_name} = {val}: {str(e)}")
            continue
    
    # Создаем DataFrame и визуализируем
    df = pd.DataFrame(results, columns=[param_name, 'N'])
    fig = px.scatter(df, x=param_name, y='N', 
                     title=f"Бифуркационная диаграмма ({param_name})", 
                     opacity=0.3,
                     labels={param_name: param_name, 
                             'N': 'Численность популяции'})
    
    fig.update_layout(
        xaxis_title=param_name,
        yaxis_title="Численность популяции",
        height=600,
        width=800
    )
    
    return fig, df

def bifurcation_diagram_stochastic(base_sim, param_name, param_range, steps, T_sim=100, 
                                  discard=50, repeats=10, sigma=0.1, current_params=None):
    """Построение бифуркационной диаграммы для стохастической модели"""
    param_values = np.linspace(param_range[0], param_range[1], steps)
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Преобразуем переведенное имя параметра в техническое
    tech_param_name = REVERSE_TRANSLATIONS.get(param_name, param_name)
    
    for i, val in enumerate(param_values):
        # Обновляем прогресс
        progress = (i + 1) / steps
        progress_bar.progress(progress)
        status_text.text(f"Прогресс: {progress:.1%} ({param_name} = {val:.2f})")
        
        # Определяем, какой параметр меняем (используя техническое имя)
        if tech_param_name == "sigma":
            sigma_val = val
            r_val = current_params.get("r", 0.1)
            K_val = current_params.get("K", 100.0)
            N0_val = current_params.get("N0", 10.0)
        elif tech_param_name == "r":
            sigma_val = current_params.get("sigma", 0.1)
            r_val = val
            K_val = current_params.get("K", 100.0)
            N0_val = current_params.get("N0", 10.0)
        elif tech_param_name == "K":
            sigma_val = current_params.get("sigma", 0.1)
            r_val = current_params.get("r", 0.1)
            K_val = val
            N0_val = current_params.get("N0", 10.0)
        elif tech_param_name == "N0":
            sigma_val = current_params.get("sigma", 0.1)
            r_val = current_params.get("r", 0.1)
            K_val = current_params.get("K", 100.0)
            N0_val = val
        else:
            # Значения по умолчанию
            sigma_val = current_params.get("sigma", 0.1)
            r_val = current_params.get("r", 0.1)
            K_val = current_params.get("K", 100.0)
            N0_val = current_params.get("N0", 10.0)
        
        # Запускаем симуляцию с позиционными аргументами
        try:
            trajectories = simulate_stochastic(
                base_sim, N0_val, r_val, K_val, T_sim,
                sigma=sigma_val, repeats=repeats
            )
            
            # Для каждой траектории берем только установившийся режим
            for traj in trajectories:
                steady_state = traj[discard:]
                # Добавляем точки в результат (берем каждую 5-ю точку для уменьшения объема)
                for x in steady_state[::5]:
                    results.append((val, x))
        except Exception as e:
            st.error(f"Ошибка при параметре {param_name} = {val}: {str(e)}")
            continue
    
    # Создаем DataFrame и визуализируем
    df = pd.DataFrame(results, columns=[param_name, 'N'])
    fig = px.scatter(df, x=param_name, y='N', 
                     title=f"Бифуркационная диаграмма ({param_name})", 
                     opacity=0.3,
                     labels={param_name: param_name, 
                             'N': 'Численность популяции'})
    
    fig.update_layout(
        xaxis_title=param_name,
        yaxis_title="Численность популяции",
        height=600,
        width=800
    )
    
    return fig, df

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
    
    # Преобразуем переведенное имя параметра в техническое
    tech_param_to_vary = REVERSE_TRANSLATIONS.get(param_to_vary, param_to_vary)
    
    if model == "Логистический рост" or model == "Модель Рикера":
        params = {
            "N0": config_params[0][0],
            "r": config_params[0][1],
            "K": config_params[0][2],
            "T": T
        }
        if tech_param_to_vary in params:
            del params[tech_param_to_vary]
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
        if tech_param_to_vary in params:
            del params[tech_param_to_vary]
        return params
    
    elif model == "Модель Лесли":
        params = {
            "N0_vec": N0_vec,
            "fertility": fertility,
            "survival": survival,
            "T": T
        }
        # Просто удаляем параметр целиком
        if tech_param_to_vary in params:
            del params[tech_param_to_vary]
        return params
    
    elif model == "Модель с задержкой":
        params = {
            "N0": common['N0'],
            "r": common['r'],
            "K": common['K'],
            "T": T,
            "tau": tau_values[0]
        }
        if tech_param_to_vary in params:
            del params[tech_param_to_vary]
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
st.set_page_config(page_title="Симулятор Популяционной Динамики", layout="wide")
st.title("🌱 Симулятор Популяционной Динамики")

# Информация в сайдбаре
st.sidebar.info("Выберите модель и настройте параметры ниже.")

# Выбор модели
model = st.sidebar.selectbox("Выберите модель:", list(model_info.keys()))
st.sidebar.caption(model_info[model])

# Общие параметры
st.sidebar.markdown("### Общие параметры")
T = st.sidebar.number_input("Количество временных шагов", min_value=1, max_value=500, value=100)


# Инициализация общих параметров
common = {}
if model != "Модель Лесли":
    common['N0'] = st.sidebar.number_input(
        "Начальная численность популяции",
        min_value=0.0,
        value=10.0,
        help="Количество особей в начальный момент времени"
    )
    common['r'] = st.sidebar.number_input(
        "Темп роста популяции",
        min_value=0.0,
        value=0.1,
        help="Скорость увеличения численности популяции"
    )
    common['K'] = st.sidebar.number_input(
        "Предельная численность среды",
        min_value=1.0,
        value=100.0,
        help="Максимальное количество особей, которое может поддерживать среда"
    )

# Специфические параметры моделей
if model == "Модель с задержкой":
    tau_values = st.sidebar.multiselect(
        "Значения временной задержки",
        options=list(range(1, 11)),
        default=[1, 2],
        help="Выберите одно или несколько значений задержки"
    )

elif model == "Гибридная модель":
    n = st.sidebar.number_input(
        "Число возрастных групп",
        min_value=1,
        max_value=10,
        value=3,
        help="Например: 3 группы = молодые/взрослые/старые особи"
    )
    
    st.sidebar.markdown("### Настройки модели")
    # Словарь с описаниями для подсказок
    feature_descriptions = {
        "Плотностная зависимость рождаемости": "Влияние численности на рождаемость",
        "Плотностная зависимость выживаемости": "Влияние численности на выживаемость",
        "Задержки рождаемости": "Учет временных задержек в процессе размножения",
        "Задержки выживаемости": "Учет временных задержек в процессе выживания",
        "Миграция между группами": "Перемещение особей между возрастными группами",
        "Случайные колебания": "Учет случайных изменений в популяции",
        "Влияние среды": "Учет периодических изменений условий среды"
    }
    model_features = {
        "Плотностная зависимость рождаемости": True,
        "Плотностная зависимость выживаемости": True,
        "Задержки рождаемости": True,
        "Задержки выживаемости": True,
        "Миграция между группами": True,
        "Случайные колебания": True,
        "Влияние среды": True
    }

    # Создание переключателей с подсказками
    for feature in model_features.keys():
        model_features[feature] = st.sidebar.toggle(
            feature,
            value=model_features[feature],
            help=feature_descriptions[feature]
        )
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
        with st.sidebar.expander("Параметры рождаемости"):
            st.markdown("""
                <div style="color: #666; font-size:0.9rem; margin-bottom:10px;">
                Укажите коэффициенты рождаемости для каждой группы
                </div>
                """, unsafe_allow_html=True)
            fert_base = [
                st.number_input(
                    f"👶 Рождаемость в группе {i + 1}",
                    min_value=0.0,
                    value=0.5,
                    help=f"Среднее число потомков на одну особь в группе {i + 1}"
                ) for i in range(n)
            ]
    else:
        fert_base = [0.0] * n


    if model_features["Плотностная зависимость выживаемости"]:
        with st.sidebar.expander("Параметры выживаемости"):
            st.markdown("""
                <div style="color: #666; font-size:0.9rem; margin-bottom:10px;">
                Укажите вероятности перехода между группами
                </div>
                """, unsafe_allow_html=True)
            surv_base = [
                st.number_input(
                    f"🔄 Переход из группы {i + 1} в группу {i + 2}",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.8,
                    help=f"Доля особей, переходящих в следующую возрастную группу"
                ) for i in range(n - 1)
            ]
    else:
        surv_base = [0.0] * (n - 1)


    if model_features["Задержки рождаемости"]:
        with st.sidebar.expander("Задержки в рождаемости"):
            st.markdown("""
                <div style="color: #666; font-size:0.9rem; margin-bottom:10px;">
                Укажите временные задержки для процессов размножения
                </div>
                """, unsafe_allow_html=True)
            delay_fert = [
                st.number_input(
                    f"⏳ Задержка для группы {i + 1}",
                    min_value=0,
                    max_value=5,
                    value=1,
                    help="Количество шагов до проявления эффекта"
                ) for i in range(n)
            ]
    else:
        delay_fert = [0] * n

    if model_features["Задержки выживаемости"]:
        with st.sidebar.expander("Задержки в выживаемости"):
            st.markdown("""
                <div style="color: #666; font-size:0.9rem; margin-bottom:10px;">
                Укажите временные задержки для процессов выживания
                </div>
                """, unsafe_allow_html=True)
            delay_surv = [
                st.number_input(
                    f"⏳ Задержка перехода {i + 1}→{i + 2}",
                    min_value=0,
                    max_value=5,
                    value=1,
                    help="Количество шагов до проявления эффекта"
                ) for i in range(n - 1)
            ]
    else:
        delay_surv = [0] * (n - 1)

    if model_features["Миграция между группами"]:
        with st.sidebar.expander("Параметры миграции"):
            st.markdown("""
                <div style="color: #666; font-size:0.9rem; margin-bottom:10px;">
                Укажите интенсивность перемещения между группами
                </div>
                """, unsafe_allow_html=True)
            migration_rates = [
                st.number_input(
                    f"🔄 Миграция из группы {i + 1}",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.1,
                    help="Доля особей, покидающих группу"
                ) for i in range(n)
            ]
    else:
        migration_rates = [0.0] * n

    st.sidebar.markdown("---")
    K = st.sidebar.number_input(
        "Максимальная численность популяции",
        min_value=1.0,
        value=100.0,
        help="Предельная емкость среды"
    )
    
    r = st.sidebar.number_input(
        "Базовый темп роста",
        min_value=0.0,
        value=0.1,
        help="Скорость роста популяции в оптимальных условиях"
    )
    
    r_surv = st.sidebar.number_input(
        "Коэффициент регуляции выживаемости",
        min_value=0.0,
        value=0.05,
        help="Влияние плотности на выживаемость особей"
    )
    if model_features["Влияние среды"]:
        env_effect = st.sidebar.slider(
            "Сила влияния среды",
            min_value=-1.0,
            max_value=1.0,
            value=0.2,
            help="От -1 (негативное) до +1 (позитивное влияние)"
        )
    else:
        env_effect = 0.0


    if model_features["Случайные колебания"]:
        stoch_intensity = st.sidebar.slider(
            "Сила случайных колебаний",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            help="От 0 (нет колебаний) до 1 (сильные колебания)"
        )
    else:
        stoch_intensity = 0.0

elif model == "Модель Лесли":
    n = st.sidebar.number_input(
        "Число возрастных групп",
        min_value=2,
        max_value=10,
        value=3,
        help="Количество различных возрастных классов в популяции"
    )
    
    with st.sidebar.expander("Коэффициенты рождаемости"):
        st.markdown("""
            <div style="color: #666; font-size:0.9rem; margin-bottom:10px;">
            Укажите, сколько потомков производит одна особь в каждой возрастной группе
            </div>
            """, unsafe_allow_html=True)
        fertility = [
            st.number_input(
                f"👶 Рождаемость в группе {i+1}",
                min_value=0.0,
                value=0.5,
                help=f"Среднее число потомков от одной особи возраста {i+1}"
            ) for i in range(n)
        ]
    
    with st.sidebar.expander("Вероятности выживания"):
        st.markdown("""
            <div style="color: #666; font-size:0.9rem; margin-bottom:10px;">
            Укажите вероятность перехода особей в следующую возрастную группу
            </div>
            """, unsafe_allow_html=True)
        survival = [
            st.number_input(
                f"🔄 Выживаемость при переходе {i+1}→{i+2}",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                help=f"Доля особей, доживающих до следующего возраста"
            ) for i in range(n-1)
        ]
    
    with st.sidebar.expander("Начальная численность"):
        st.markdown("""
            <div style="color: #666; font-size:0.9rem; margin-bottom:10px;">
            Укажите начальное количество особей в каждой возрастной группе
            </div>
            """, unsafe_allow_html=True)
        N0_vec = [
            st.number_input(
                f"🔢 Начальная численность группы {i+1}",
                min_value=0.0,
                value=10.0,
                help=f"Количество особей возраста {i+1} в начальный момент"
            ) for i in range(n)
        ]


elif model == "Стохастическая симуляция":
    repeats = st.sidebar.number_input(
        "Количество повторений",
        min_value=1,
        max_value=200,
        value=50,
        help="Сколько раз повторить симуляцию для учета случайности"
    )
    
    sigma_values = st.sidebar.multiselect(
        "Уровни случайных колебаний",
        options=[0.0, 0.05, 0.1, 0.2, 0.5],
        default=[0.1],
        help="Выберите силу случайных воздействий (σ)"
    )
    
    base_model = st.sidebar.selectbox(
        "Базовая модель",
        ["Логистический рост", "Модель Рикера"],
        help="Выберите тип роста, к которому добавляются случайные колебания"
    )
    
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
# Добавляем в сайдбар выбор параметра для бифуркационного анализа
with st.sidebar.expander("Настройки бифуркационного анализа", expanded=True):
    enable_bifurcation = st.checkbox("Включить бифуркационный анализ", value=True)
    
    if enable_bifurcation:
        # Получаем конфигурацию для выбранной модели
        model_config = HEATMAP_PARAM_CONFIGS[model]
        
        # Выбор параметра для анализа
        bifurcation_param = st.selectbox(
            "Параметр для бифуркационного анализа",
            options=model_config["params"],
            key="bifurcation_param"
        )
        
        # Настройка диапазона значений
        st.markdown("### Диапазон значений параметра")
        
        if bifurcation_param in model_config.get("multi_param", []):
            st.markdown(f"**Примечание**: Для параметра '{bifurcation_param}' будет использовано одно значение для всех элементов")
            default_range = model_config["ranges"][bifurcation_param]
            bifurcation_range = st.slider(
                f"Диапазон {bifurcation_param}",
                min_value=float(default_range[0]),
                max_value=float(default_range[1]),
                value=(float(default_range[0]), float(default_range[1])),
                step=0.1,
                key="bifurcation_range"
            )
        else:
            # Для обычных параметров
            default_range = model_config["ranges"][bifurcation_param]
            if bifurcation_param == "tau":
                bifurcation_range = st.slider(
                    f"Диапазон {bifurcation_param}",
                    min_value=int(default_range[0]),
                    max_value=int(default_range[1]),
                    value=(int(default_range[0]), int(default_range[1])),
                    step=1,
                    key="bifurcation_range_tau"
                )
            else:
                bifurcation_range = st.slider(
                    f"Диапазон {bifurcation_param}",
                    min_value=float(default_range[0]),
                    max_value=float(default_range[1]),
                    value=(float(default_range[0]), float(default_range[1])),
                    step=0.1,
                    key="bifurcation_range_std"
                )
        
        # Настройки разрешения
        st.markdown("### Настройки разрешения")
        bifurcation_steps = st.slider(
            "Количество шагов параметра",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            key="bifurcation_steps"
        )
        
        bifurcation_discard = st.slider(
            "Отбросить начальные шаги",
            min_value=0,
            max_value=100,
            value=50,
            step=10,
            key="bifurcation_discard"
        )
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
                st.subheader("Фазовый портрет логистической модели")
                plot_and_analyze_phase_portrait(traj, "Логистический рост", delays=[1, 2], include_3d=True)

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
                st.subheader("Фазовые портреты для разных конфигураций")
                for idx, (config_name, traj) in enumerate(all_trajs.items()):
                    with st.expander(f"Фазовый портрет для {config_name}", expanded=(idx == 0)):
                        plot_and_analyze_phase_portrait(traj, config_name, delays=[1, 2], include_3d=(idx == 0))
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
            #plot_phase_portrait(history[:, 0])  # первая возрастная группа
            st.subheader("Фазовые портреты гибридной модели")
            st.markdown("#### Фазовый портрет общей численности")
            plot_and_analyze_phase_portrait(total_pop, "Общая численность", delays=[1, 2], include_3d=True)
            
            # Фазовые портреты по возрастным группам
            with st.expander("Фазовые портреты по возрастным группам"):
                for i in range(n):
                    st.markdown(f"#### Возрастная группа {i+1}")
                    plot_and_analyze_phase_portrait(history[:, i], f"Возрастная группа {i+1}", delays=[1], include_3d=False)

            if enable_bifurcation and model == "Гибридная модель":
                st.subheader("Бифуркационный анализ")
                with st.spinner("Построение бифуркационной диаграммы..."):
                    current_params = collect_current_params(model, bifurcation_param)
                    current_params["T"] = T
                    
                    # Специальная обработка для параметров гибридной модели
                    if bifurcation_param == "migration_rates":
                        current_params["migration_rates"] = [0.0] * n  # Будет заменено в функции
                    elif bifurcation_param == "delay_fert":
                        current_params["delay_fert"] = [0] * n  # Будет заменено в функции
                    elif bifurcation_param == "delay_surv":
                        current_params["delay_surv"] = [0] * (n - 1)  # Будет заменено в функции
                    
                    fig_bifurcation, df_bifurcation = bifurcation_diagram(
                        model_func=simulate_hybrid,
                        param_name=bifurcation_param,
                        param_range=bifurcation_range,
                        steps=bifurcation_steps,
                        T_sim=T,
                        discard=bifurcation_discard,
                        current_params=current_params
                    )
                    st.plotly_chart(fig_bifurcation)
                    
                    with st.expander("Анализ бифуркационной диаграммы"):
                        st.write("""
                        **Интерпретация бифуркационной диаграммы для гибридной модели:**
                        
                        Гибридная модель объединяет несколько механизмов и может демонстрировать сложное поведение:
                        
                        - **Стационарные состояния**: Устойчивые точки равновесия
                        - **Периодические колебания**: Регулярные циклы разной длительности
                        - **Квазипериодические режимы**: Нерегулярные, но ограниченные колебания
                        - **Хаотические режимы**: Непредсказуемое поведение с высокой чувствительностью к начальным условиям
                        
                        Особое влияние на динамику оказывают:
                        - Задержки (могут вызывать колебания)
                        - Миграция (может стабилизировать или дестабилизировать систему)
                        - Плотностная зависимость (может приводить к хаотическим режимам)
                        """)
                        
                        # Анализ чувствительности к параметру
                        sensitivity = np.std(df_bifurcation.groupby(bifurcation_param)['N'].std())
                        st.metric("Чувствительность к параметру", f"{sensitivity:.3f}")

            
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
                st.subheader("Фазовый портрет модели Рикера")
                plot_and_analyze_phase_portrait(traj, "Модель Рикера", delays=[1, 2], include_3d=True)

                if enable_bifurcation:
                    st.subheader("Бифуркационный анализ")
                    with st.spinner("Построение бифуркационной диаграммы..."):
                        current_params = collect_current_params(model, bifurcation_param)
                        current_params["T"] = T
                        
                        fig_bifurcation, df_bifurcation = bifurcation_diagram(
                            model_func=simulate_ricker,
                            param_name=bifurcation_param,
                            param_range=bifurcation_range,
                            steps=bifurcation_steps,
                            T_sim=T,
                            discard=bifurcation_discard,
                            current_params=current_params
                        )
                        st.plotly_chart(fig_bifurcation)
                        
                        with st.expander("Анализ бифуркационной диаграммы"):
                            st.write("""
                            **Интерпретация бифуркационной диаграммы:**
                            
                            Модель Рикера известна своими богатыми динамическими режимами:
                            - При малых значениях r: стационарное состояние
                            - При средних значениях r: колебания с периодом 2, затем 4, 8...
                            - При больших значениях r: хаотический режим
                            
                            Это классический пример каскада бифуркаций удвоения периода.
                            """)
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
                st.subheader("Фазовые портреты для разных конфигураций")
                for idx, (config_name, traj) in enumerate(all_trajs.items()):
                    with st.expander(f"Фазовый портрет для {config_name}", expanded=(idx == 0)):
                        plot_and_analyze_phase_portrait(traj, config_name, delays=[1, 2], include_3d=(idx == 0))
                
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
                if enable_bifurcation and model == "Модель с задержкой":
                    st.subheader("Бифуркационный анализ")
                    with st.spinner("Построение бифуркационной диаграммы..."):
                        current_params = collect_current_params(model, bifurcation_param)
                        current_params["T"] = T
                        
                        # Убедимся, что параметр tau присутствует
                        if "tau" not in current_params:
                            current_params["tau"] = tau_values[0] if tau_values else 1
                        
                        fig_bifurcation, df_bifurcation = bifurcation_diagram(
                            model_func=simulate_delay,
                            param_name=bifurcation_param,
                            param_range=bifurcation_range,
                            steps=bifurcation_steps,
                            T_sim=T,
                            discard=bifurcation_discard,
                            current_params=current_params
                        )
                        st.plotly_chart(fig_bifurcation)
                        
                        with st.expander("Анализ бифуркационной диаграммы"):
                            st.write("""
                            **Интерпретация бифуркационной диаграммы для модели с задержкой:**
                            
                            Модели с временной задержкой демонстрируют особенно богатую динамику:
                            
                            - При малых задержках система может вести себя как обычная логистическая модель
                            - При увеличении задержки появляются колебания
                            - При критических значениях задержки возникают бифуркации
                            - Большие задержки часто приводят к хаотическому поведению
                            
                            Задержка действует как дестабилизирующий фактор, так как система реагирует на состояние в прошлом, а не на текущее состояние.
                            """)
                            
                            # Анализ критических значений
                            if bifurcation_param == "tau":
                                # Ищем точку первой бифуркации
                                grouped = df_bifurcation.groupby(bifurcation_param)
                                variance = grouped['N'].var()
                                # Находим первое значение tau, где дисперсия значительно возрастает
                                threshold = variance.mean() * 0.1
                                critical_points = variance[variance > threshold].index.tolist()
                                
                                if critical_points:
                                    first_critical = critical_points[0]
                                    st.metric("Критическое значение задержки", f"{first_critical:.2f}")
                                    st.write(f"При задержке τ ≈ {first_critical:.2f} система теряет устойчивость и начинаются колебания.")
                            
                            elif bifurcation_param == "r":
                                # Анализ влияния темпа роста при фиксированной задержке
                                st.write(f"Анализ при фиксированной задержке τ = {current_params['tau']}")
                                st.write("""
                                При увеличении темпа роста r в модели с задержкой:
                                - Малые значения r: стабильное равновесие
                                - Средние значения r: колебания с нарастающей амплитудой
                                - Большие значения r: сложная динамика, возможен хаос
                                """)
                st.subheader("Фазовые портреты для модели с задержкой")
                for tau_i, traj in all_trajs.items():
                    with st.expander(f"Фазовый портрет для {tau_i}", expanded=(tau_i == f"τ = {tau_values[0]}")):
                        plot_and_analyze_phase_portrait(traj, f"Модель с задержкой {tau_i}", 
                                                    delays=[1, min(5, int(tau_i.split('=')[1].strip()))])
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
                        # Добавьте после анализа матрицы Лесли:
                    st.subheader("Фазовые портреты модели Лесли")
                    
                    # Общий фазовый портрет
                    st.markdown("#### Фазовый портрет общей численности")
                    plot_and_analyze_phase_portrait(total_pop, "Модель Лесли (общая численность)", delays=[1, 2], include_3d=True)
                    
                    # Фазовые портреты по возрастным группам
                    st.subheader("Фазовые портреты по возрастным группам")
                    for i in range(n):
                        st.markdown(f"#### Возрастная группа {i+1}")
                        plot_and_analyze_phase_portrait(history[:, i], f"Возрастная группа {i+1}", delays=[1], include_3d=False)

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
            if enable_bifurcation and model == "Модель Лесли":
                st.subheader("Бифуркационный анализ")
                with st.spinner("Построение бифуркационной диаграммы..."):
                    current_params = collect_current_params(model, bifurcation_param)
                    current_params["T"] = T
                    
                    # Специальная обработка для параметров модели Лесли
                    if bifurcation_param == "fertility":
                        current_params["fertility"] = [0.0] * n  # Будет заменено в функции
                    elif bifurcation_param == "survival":
                        current_params["survival"] = [0.0] * (n - 1)  # Будет заменено в функции
                    elif bifurcation_param == "N0_vec":
                        current_params["N0_vec"] = [0.0] * n  # Будет заменено в функции
                    
                    fig_bifurcation, df_bifurcation = bifurcation_diagram(
                        model_func=simulate_leslie,
                        param_name=bifurcation_param,
                        param_range=bifurcation_range,
                        steps=bifurcation_steps,
                        T_sim=T,
                        discard=bifurcation_discard,
                        current_params=current_params
                    )
                    st.plotly_chart(fig_bifurcation)
                    
                    with st.expander("Анализ бифуркационной диаграммы"):
                        st.write("""
                        **Интерпретация бифуркационной диаграммы для модели Лесли:**
                        
                        Модель Лесли описывает возрастно-структурированную популяцию:
                        
                        - Динамика определяется собственными значениями матрицы Лесли
                        - Если доминирующее собственное значение > 1: популяция растет
                        - Если доминирующее собственное значение < 1: популяция сокращается
                        - Если доминирующее собственное значение = 1: стабильная популяция
                        
                        Бифуркационная диаграмма показывает, как изменение параметра влияет на установившуюся численность популяции.
                        """)
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
                    # ИСПРАВЛЕННЫЙ ВЫЗОВ - передаем позиционные аргументы
                    results = simulate_stochastic(
                        base_sim,
                        common['N0'],  # позиционный аргумент
                        common['r'],   # позиционный аргумент
                        common['K'],   # позиционный аргумент
                        T,             # позиционный аргумент
                        sigma=sigma,   # именованный аргумент
                        repeats=repeats # именованный аргумент
                    )
                    
                    # Остальной код остается без изменений...
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
                st.subheader("Фазовые портреты стохастической модели")
                for sigma, mean_traj in all_means.items():
                        with st.expander(f"Фазовый портрет для {sigma}", expanded=(sigma == f"σ={sigma_values[0]}")):
                            plot_and_analyze_phase_portrait(mean_traj, f"Средняя траектория ({sigma})", delays=[1, 2], include_3d=True)
                    
                    # Дополнительно: фазовый портрет для одной случайной реализации
                if len(results) > 0:
                    with st.expander("Фазовый портрет для отдельной реализации"):
                        random_idx = np.random.randint(0, len(results))
                        plot_and_analyze_phase_portrait(results[random_idx], 
                                                        f"Случайная реализация (σ={sigma_values[0]})", 
                                                        delays=[1, 2], 
                                                        include_3d=True)
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
                if enable_bifurcation and model == "Стохастическая симуляция":
                    st.subheader("Бифуркационный анализ")
                    with st.spinner("Построение бифуркационной диаграммы..."):
                        current_params = collect_current_params(model, bifurcation_param)
                        
                        # Для стохастической модели используем специальную функцию
                        fig_bifurcation, df_bifurcation = bifurcation_diagram_stochastic(
                            base_sim=base_sim,
                            param_name=bifurcation_param,
                            param_range=bifurcation_range,
                            steps=bifurcation_steps,
                            T_sim=T,
                            discard=bifurcation_discard,
                            repeats=min(repeats, 20),  # Ограничиваем количество повторений для ускорения
                            sigma=sigma_values[0] if sigma_values else 0.1,
                            current_params=current_params
                        )
                        st.plotly_chart(fig_bifurcation)
                        
                        with st.expander("Анализ бифуркационной диаграммы"):
                            st.write("""
                            **Интерпретация бифуркационной диаграммы для стохастической модели:**
                            
                            В стохастической модели:
                            
                            - Вместо четких бифуркаций наблюдается размытие точек
                            - Шум может как стабилизировать, так и дестабилизировать систему
                            - При малых значениях шума (σ) видна базовая структура детерминистической модели
                            - При больших значениях шума структура размывается
                            
                            Важно отметить, что каждая реализация стохастической модели уникальна, поэтому диаграмма показывает распределение возможных состояний.
                            """)
                            
                            # Анализ разброса
                            variance_by_param = df_bifurcation.groupby(bifurcation_param)['N'].var()
                            mean_variance = variance_by_param.mean()
                            st.metric("Средняя дисперсия", f"{mean_variance:.3f}")
                            
                            # Визуализация разброса
                            variance_df = pd.DataFrame({
                                bifurcation_param: variance_by_param.index,
                                'Дисперсия': variance_by_param.values
                            })
                            fig_var = px.line(variance_df, x=bifurcation_param, y='Дисперсия', 
                                            title=f"Зависимость дисперсии от параметра {bifurcation_param}")
                            st.plotly_chart(fig_var)
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
st.sidebar.info("Разработано Лией Ахметовой — v1.0")
