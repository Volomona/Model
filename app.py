import sys
import io
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Проверка наличия Streamlit
try:
    import streamlit as st
except ModuleNotFoundError:
    sys.exit("Ошибка: Streamlit не установлен. Установите и запустите: `streamlit run app.py`")

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================== ФУНКЦИИ СИМУЛЯЦИИ С КЭШИРОВАНИЕМ ================== #

@st.cache_data(max_entries=5)
def simulate_logistic(N0: float, r: float, K: float, T: int) -> np.ndarray:
    """Логистический рост с обработкой ошибок"""
    Ns = np.zeros(T + 1)
    Ns[0] = N0
    try:
        for t in range(T):
            next_N = Ns[t] + r * Ns[t] * (1 - Ns[t] / K)
            if next_N < 0 or np.isnan(next_N):
                return Ns[:t+1]
            Ns[t+1] = next_N
        return Ns
    except Exception as e:
        logger.error(f"Ошибка логистической симуляции: {str(e)}")
        return np.array([N0])

@st.cache_data(max_entries=5)
def simulate_ricker(N0: float, r: float, K: float, T: int) -> np.ndarray:
    """Модель Рикера с обработкой ошибок"""
    Ns = np.zeros(T + 1)
    Ns[0] = N0
    try:
        for t in range(T):
            next_N = Ns[t] * np.exp(r * (1 - Ns[t] / K))
            if next_N < 0 or np.isnan(next_N):
                return Ns[:t+1]
            Ns[t+1] = next_N
        return Ns
    except Exception as e:
        logger.error(f"Ошибка модели Рикера: {str(e)}")
        return np.array([N0])

@st.cache_data(max_entries=5)
def simulate_leslie(N0_vec: list, fertility: list, survival: list, T: int) -> np.ndarray:
    """Матричная модель Лесли с валидацией"""
    try:
        validate_leslie_params(survival, fertility)
        
        n = len(N0_vec)
        N = np.array(N0_vec, dtype=float)
        history = [N.copy()]
        
        L = np.zeros((n, n))
        L[0, :] = fertility
        for i in range(1, n):
            L[i, i-1] = survival[i-1]
            
        for _ in range(T):
            N = L.dot(N)
            N = np.clip(N, 0, None)  # Защита от отрицательных значений
            history.append(N.copy())
        return np.array(history)
    except Exception as e:
        logger.error(f"Ошибка модели Лесли: {str(e)}")
        return np.array([N0_vec])

@st.cache_data(max_entries=5)
def simulate_delay(N0: float, r: float, K: float, T: int, tau: int) -> np.ndarray:
    """Модель с запаздыванием"""
    try:
        Ns = np.zeros(T + tau + 1)
        Ns[:tau+1] = N0
        for t in range(tau, T + tau):
            next_N = Ns[t] * np.exp(r * (1 - Ns[t - tau] / K))
            if next_N < 0 or np.isnan(next_N):
                return Ns[:t+1]
            Ns[t+1] = next_N
        return Ns[tau:]  # Возвращаем только релевантные значения
    except Exception as e:
        logger.error(f"Ошибка модели с запаздыванием: {str(e)}")
        return np.array([N0])

def simulate_stochastic(base_sim, *args, sigma: float = 0.1, repeats: int = 100) -> np.ndarray:
    """Стохастическая симуляция с векторизацией"""
    try:
        base_traj = base_sim(*args)
        repeats = min(repeats, 500)  # Ограничение для защиты
        
        # Векторизованные вычисления
        noise = np.random.normal(0, sigma, size=(repeats, len(base_traj)))
        results = np.clip(base_traj + noise, a_min=0, a_max=None)
        
        return results
    except Exception as e:
        logger.error(f"Ошибка стохастической симуляции: {str(e)}")
        return np.array([base_sim(*args)])

# ================== ВАЛИДАЦИЯ ================== #

def validate_leslie_params(survival: list, fertility: list):
    """Проверка параметров модели Лесли"""
    if sum(survival) > 1.0:
        raise ValueError("Сумма вероятностей выживания > 1.0")
    if any(f < 0 for f in fertility):
        raise ValueError("Коэффициенты фертильности < 0")
    if any(s < 0 or s > 1 for s in survival):
        raise ValueError("Вероятности выживания вне [0,1]")

# ================== ВИЗУАЛИЗАЦИЯ ================== #

def plot_and_export(data, title, log_scale=False):
    """Улучшенная визуализация с экспортом"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if data.ndim == 1:
        ax.plot(data, label='Основная траектория', linewidth=2)
    else:
        for i in range(min(data.shape[0], 100)):  # Ограничение числа линий
            ax.plot(data[i], alpha=0.1, color='blue', 
                   label='Стохастические траектории' if i == 0 else "")
        ax.plot(np.nanmean(data, axis=0), color='red', 
               linewidth=2, label='Среднее значение')
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Временной шаг', fontsize=12)
    ax.set_ylabel('Размер популяции', fontsize=12)
    
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-1)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    st.pyplot(fig)
    
    # Экспорт PNG
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    st.download_button(
        "Скачать график (PNG)", 
        data=buf, 
        file_name=f"{title.replace(' ', '_')}.png", 
        mime="image/png"
    )
    
    # Экспорт CSV для числовых данных
    if data.ndim == 1:
        df = pd.DataFrame(data, columns=['Population'])
    else:
        df = pd.DataFrame(data.T)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Скачать данные (CSV)", 
        data=csv, 
        file_name=f"{title.replace(' ', '_')}.csv", 
        mime="text/csv"
    )

# ================== ПОЛЬЗОВАТЕЛЬСКИЙ ИНТЕРФЕЙС ================== #

st.set_page_config(
    page_title="Симулятор Популяционной Динамики", 
    layout="wide",
    page_icon="🌱"
)
st.title("🌱 Симулятор Популяционной Динамики")

# Информация о моделях
model_info = {
    "Логистический рост": "Классическая логистическая модель с емкостью K",
    "Модель Рикера": "Экспоненциальный рост с зависимостью от плотности",
    "Матрица Лесли": "Возрастно-структурированная модель",
    "Модель с запаздыванием": "Зависимость от состояния в прошлом",
    "Стохастическая": "Многократные запуски с шумом"
}

# Боковая панель
st.sidebar.header("Настройки модели")
model = st.sidebar.selectbox(
    "Выберите модель:", 
    list(model_info.keys()),
    help="Выберите тип популяционной модели"
)
st.sidebar.markdown(f"**Описание:** {model_info[model]}")

# Общие параметры
st.sidebar.subheader("Основные параметры")
T = st.sidebar.slider(
    "Временные шаги (T)", 
    min_value=1, max_value=500, value=100,
    help="Количество шагов симуляции"
)

common_params = {}
if model != "Матрица Лесли":
    common_params['N0'] = st.sidebar.number_input(
        "Начальная популяция (N0)", 
        min_value=0.0, value=10.0, step=1.0
    )
    common_params['r'] = st.sidebar.number_input(
        "Скорость роста (r)", 
        min_value=0.0, value=0.5, step=0.1,
        help="Базовый коэффициент роста"
    )
    common_params['K'] = st.sidebar.number_input(
        "Емкость среды (K)", 
        min_value=1.0, value=100.0, step=10.0,
        help="Максимальная поддерживаемая популяция"
    )
    
    if model == "Логистический рост" and common_params.get('r', 0) > 3.0:
        st.sidebar.warning(
            "Высокий коэффициент роста (r > 3) может вызывать хаотическое поведение!"
        )

# Специфические параметры моделей
if model == "Модель с запаздыванием":
    tau = st.sidebar.slider(
        "Запаздывание (τ)", 
        min_value=1, max_value=20, value=5,
        help="Временная задержка влияния на рост"
    )

elif model == "Матрица Лесли":
    n_age = st.sidebar.slider(
        "Количество возрастных классов", 
        min_value=2, max_value=10, value=3
    )
    
    st.sidebar.subheader("Параметры Лесли")
    with st.sidebar.expander("Коэффициенты фертильности"):
        fertility = [
            st.number_input(f"Фертильность класса {i}", 
                          min_value=0.0, value=0.5, step=0.1)
            for i in range(n_age)
        ]
    
    with st.sidebar.expander("Вероятности выживания"):
        survival = [
            st.number_input(f"Выживаемость {i}→{i+1}", 
                          min_value=0.0, max_value=1.0, value=0.8, step=0.05)
            for i in range(n_age-1)
        ]
    
    with st.sidebar.expander("Начальная популяция"):
        N0_vec = [
            st.number_input(f"Класс {i}", min_value=0.0, value=10.0, step=1.0)
            for i in range(n_age)
        ]

elif model == "Стохастическая":
    st.sidebar.subheader("Стохастические параметры")
    base_model = st.sidebar.selectbox(
        "Базовая модель", 
        ["Логистический рост", "Модель Рикера"]
    )
    repeats = st.sidebar.slider(
        "Количество повторов", 
        min_value=10, max_value=500, value=100
    )
    sigma = st.sidebar.slider(
        "Уровень шума (σ)", 
        min_value=0.0, max_value=1.0, value=0.2, step=0.05
    )

# Дополнительные настройки
st.sidebar.subheader("Настройки отображения")
log_scale = st.sidebar.checkbox(
    "Логарифмическая шкала", 
    help="Использовать логарифмическую шкалу для оси Y"
)
show_stats = st.sidebar.checkbox(
    "Показать статистику", 
    value=True,
    help="Отображать статистические характеристики"
)

# ================== ЗАПУСК СИМУЛЯЦИИ ================== #

if st.sidebar.button("Запустить симуляцию", type="primary"):
    with st.spinner("Выполняется симуляция..."):
        try:
            if model == "Логистический рост":
                traj = simulate_logistic(**common_params, T=T)
                plot_and_export(traj, "Логистический рост", log_scale)
                
            elif model == "Модель Рикера":
                traj = simulate_ricker(**common_params, T=T)
                plot_and_export(traj, "Модель Рикера", log_scale)
                
            elif model == "Матрица Лесли":
                traj = simulate_leslie(N0_vec, fertility, survival, T)
                df = pd.DataFrame(
                    traj, 
                    columns=[f"Возраст {i}" for i in range(n_age)]
                )
                st.line_chart(df)
                
                # Анализ матрицы Лесли
                L = np.zeros((n_age, n_age))
                L[0, :] = fertility
                for i in range(1, n_age):
                    L[i, i-1] = survival[i-1]
                
                eigvals = np.linalg.eigvals(L)
                dominant = np.max(np.real(eigvals))
                st.write(f"**Собственное число:** {dominant:.3f}")
                
                if dominant > 1:
                    st.success("Популяция растет (λ > 1)")
                elif dominant < 1:
                    st.error("Популяция сокращается (λ < 1)")
                else:
                    st.info("Стабильная популяция (λ ≈ 1)")
                
            elif model == "Модель с запаздыванием":
                traj = simulate_delay(**common_params, T=T, tau=tau)
                plot_and_export(traj, "Модель с запаздыванием", log_scale)
                
            elif model == "Стохастическая":
                base_fn = simulate_logistic if base_model == "Логистический рост" else simulate_ricker
                results = simulate_stochastic(
                    base_fn, 
                    common_params['N0'], 
                    common_params['r'], 
                    common_params['K'], 
                    T,
                    sigma=sigma,
                    repeats=repeats
                )
                
                plot_and_export(results, "Стохастическая симуляция", log_scale)
                
                if show_stats:
                    st.subheader("Статистика")
                    stats_df = pd.DataFrame({
                        'Mean': np.nanmean(results, axis=0),
                        'Std': np.nanstd(results, axis=0),
                        'Min': np.nanmin(results, axis=0),
                        'Max': np.nanmax(results, axis=0)
                    })
                    st.dataframe(stats_df.style.background_gradient(), use_container_width=True)
            
            # Проверка результатов
            if 'traj' in locals() and traj is not None:
                if np.isnan(traj).any():
                    st.warning("Обнаружены NaN значения в результатах")
                if (traj < 0).any():
                    st.warning("Обнаружены отрицательные значения популяции")
                    
        except Exception as e:
            st.error(f"Ошибка при выполнении симуляции: {str(e)}")
            logger.exception("Simulation error")

# Нижний колонтитул
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Справка:**
- Логистическая модель: Nₜ₊₁ = Nₜ + rNₜ(1-Nₜ/K)
- Модель Рикера: Nₜ₊₁ = Nₜexp[r(1-Nₜ/K)]
""")
st.sidebar.caption("v1.2 | © 2023 | Разработано с использованием Python и Streamlit")
