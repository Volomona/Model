import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def simulate_logistic(N0, r, K, T):
    Ns = [N0]
    for _ in range(T):
        Nt = Ns[-1]
        if K <= 0 or Nt > 10 * K or Nt < -10 * K:
            Ns.append(np.nan)
            continue
        next_N = Nt + r * Nt * (1 - Nt / K)
        Ns.append(next_N if np.isfinite(next_N) else np.nan)
    return np.array(Ns)

def simulate_ricker(N0, r, K, T):
    Ns = [N0]
    for _ in range(T):
        Nt = Ns[-1]
        if K <= 0 or Nt > 10 * K or Nt < -10 * K:
            Ns.append(np.nan)
            continue
        next_N = Nt * np.exp(r * (1 - Nt / K))
        Ns.append(next_N if np.isfinite(next_N) else np.nan)
    return np.array(Ns)

def simulate_leslie_lambda_max(fertility, survival):
    n = len(fertility)
    L = np.zeros((n, n))
    L[0, :] = fertility
    for i in range(1, n):
        L[i, i-1] = survival[i-1]
    eigvals = np.linalg.eigvals(L)
    return np.max(np.abs(eigvals))

def simulate_delay_amplitude(N0, r, K, T, tau, T_metric_calc):
    if tau < 1:
        tau = 1
    Ns = [N0] * (tau + 1)
    for _ in range(T):
        N_t = Ns[-1]
        N_t_minus_tau = Ns[-(tau + 1)]
        if K <= 0:
            next_N = np.nan
        else:
            next_N = N_t * np.exp(r * (1 - N_t_minus_tau / K))
        Ns.append(next_N if np.isfinite(next_N) else np.nan)

    simulated_part = np.array(Ns[tau + 1 : tau + 1 + T])
    traj_clean = simulated_part[~np.isnan(simulated_part)]
    if traj_clean.size > T_metric_calc:
        settled = traj_clean[-T_metric_calc:]
        if settled.size > 1:
            return np.max(settled) - np.min(settled)
    return np.nan

def simulate_stochastic_std(N0, r, K, T, sigma, repeats):
    finals = []
    for _ in range(repeats):
        traj = simulate_ricker(N0, r, K, T)
        noise = np.random.normal(0, sigma, size=traj.shape)
        noisy = traj + noise
        noisy = np.where(np.isfinite(noisy), noisy, np.nan)
        final_vals = noisy[~np.isnan(noisy)]
        if final_vals.size > 0:
            finals.append(final_vals[-1])
    finals = np.array(finals)
    return np.nanstd(finals) if finals.size > 1 else np.nan

N0_default = 10.0
T_sim = 200            
T_metric = 50          

# 1) Логистическая модель: r ∈ [0.5, 2.0] с шагом 0.1; K ∈ [50, 100, ..., 500]
r_log_vals = np.arange(0.5, 2.01, 0.1)
K_log_vals = np.arange(50, 501, 50)

# 2) Модель Рикера: те же диапазоны r и K
r_rick_vals = np.arange(0.5, 2.01, 0.1)
K_rick_vals = np.arange(50, 501, 50)

# 3) Модель Лесли (n=3, базовые значения):
f_base = np.array([0.5, 0.3, 0.2])
s_base = np.array([0.8, 0.6])
# Варьируем каждый параметр ±20%
f0_vals = np.linspace(f_base[0] * 0.8, f_base[0] * 1.2, 10)   # 10 точек от 0.4 до 0.6
s0_vals = np.linspace(s_base[0] * 0.8, s_base[0] * 1.2, 10)   # 10 точек от 0.64 до 0.96

# 4) Модель с запаздыванием: τ ∈ {1, 2, 5}, r ∈ [0.5, 2.0] шагом 0.1, K = 100
tau_vals = [1, 2, 5]
r_delay_vals = np.arange(0.5, 2.01, 0.1)
K_delay_fixed = 100.0

# 5) Стохастическая модель: r ∈ [0.5, 2.0] шагом 0.1, sigma ∈ {0.0, 0.1, 0.5}, K=100, repeats=100
r_stoch_vals = np.arange(0.5, 2.01, 0.1)
sigma_vals = [0.0, 0.1, 0.5]
K_stoch_fixed = 100.0
repeats_stoch = 100
T_stoch = 100   # для стохастики достаточно 100 шагов

# 1) Логистическая модель
amplitude_log = np.full((len(K_log_vals), len(r_log_vals)), np.nan)
for i, K_val in enumerate(K_log_vals):
    for j, r_val in enumerate(r_log_vals):
        traj = simulate_logistic(N0_default, r_val, K_val, T_sim)
        traj_clean = traj[~np.isnan(traj)]
        if traj_clean.size > T_metric:
            settled = traj_clean[-T_metric:]
            amplitude_log[i, j] = settled.max() - settled.min()

# 2) Модель Рикера
amplitude_rick = np.full((len(K_rick_vals), len(r_rick_vals)), np.nan)
for i, K_val in enumerate(K_rick_vals):
    for j, r_val in enumerate(r_rick_vals):
        traj = simulate_ricker(N0_default, r_val, K_val, T_sim)
        traj_clean = traj[~np.isnan(traj)]
        if traj_clean.size > T_metric:
            settled = traj_clean[-T_metric:]
            amplitude_rick[i, j] = settled.max() - settled.min()

# 3) Модель Лесли
lambda_matrix = np.full((len(s0_vals), len(f0_vals)), np.nan)
f1_fixed, f2_fixed = f_base[1], f_base[2]
s1_fixed = s_base[1]
for i, s0 in enumerate(s0_vals):
    for j, f0 in enumerate(f0_vals):
        fertility = [f0, f1_fixed, f2_fixed]
        survival = [s0, s1_fixed]
        lambda_matrix[i, j] = simulate_leslie_lambda_max(fertility, survival)

# 4) Модель с запаздыванием
amplitude_delay = np.full((len(tau_vals), len(r_delay_vals)), np.nan)
for i, tau in enumerate(tau_vals):
    for j, r_val in enumerate(r_delay_vals):
        amp = simulate_delay_amplitude(N0_default, r_val, K_delay_fixed, T_sim, tau, T_metric)
        amplitude_delay[i, j] = amp

# 5) Стохастическая модель
std_matrix = np.full((len(sigma_vals), len(r_stoch_vals)), np.nan)
for i, sigma in enumerate(sigma_vals):
    for j, r_val in enumerate(r_stoch_vals):
        std_matrix[i, j] = simulate_stochastic_std(N0_default, r_val, K_stoch_fixed, T_stoch, sigma, repeats_stoch)

def plot_heatmap(matrix, x_vals, y_vals, xlabel, ylabel, title, cmap, cbar_label, num_xticks=8, num_yticks=5):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, cmap=cmap, cbar_kws={'label': cbar_label}, 
                xticklabels=False, yticklabels=False)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)

    xticks_idx = np.linspace(0, len(x_vals) - 1, num_xticks, dtype=int)
    yticks_idx = np.linspace(0, len(y_vals) - 1, num_yticks, dtype=int)
    plt.xticks(xticks_idx + 0.5, [f"{x_vals[idx]:.1f}" for idx in xticks_idx], rotation=45, ha='right')
    plt.yticks(yticks_idx + 0.5, [f"{y_vals[idx]:.1f}" for idx in yticks_idx])

    plt.tight_layout()

# 1) Логистическая модель
plot_heatmap(
    amplitude_log, 
    x_vals=r_log_vals, 
    y_vals=K_log_vals, 
    xlabel="r (темп роста от 0.5 до 2.0)", 
    ylabel="K (ёмкость среды от 50 до 500)", 
    title="Логистическая модель: амплитуда(N) от r и K", 
    cmap="viridis", 
    cbar_label="Амплитуда"
)

# 2) Модель Рикера
plot_heatmap(
    amplitude_rick, 
    x_vals=r_rick_vals, 
    y_vals=K_rick_vals,
    xlabel="r (темп роста от 0.5 до 2.0)", 
    ylabel="K (ёмкость среды от 50 до 500)", 
    title="Модель Рикера: амплитуда(N) от r и K", 
    cmap="magma", 
    cbar_label="Амплитуда"
)

# 3) Модель Лесли
plot_heatmap(
    lambda_matrix,
    x_vals=f0_vals,
    y_vals=s0_vals,
    xlabel="f0 (рождаемость класса 0 от 0.4 до 0.6)",
    ylabel="s0 (выживаемость класса 0 от 0.64 до 0.96)",
    title="Модель Лесли (n=3): λ_max от f0 и s0",
    cmap="coolwarm",
    cbar_label="λ_max"
)

# 4) Модель с запаздыванием
plot_heatmap(
    amplitude_delay,
    x_vals=r_delay_vals,
    y_vals=tau_vals,
    xlabel="r (темп роста от 0.5 до 2.0)",
    ylabel="τ (задержка ∈ {1,2,5})",
    title=f"Модель с запаздыванием (K = {K_delay_fixed}): амплитуда(N) от r и τ",
    cmap="plasma",
    cbar_label="Амплитуда",
    num_xticks=8,
    num_yticks=len(tau_vals)
)

# 5) Стохастическая модель (на базе Рикера)
plot_heatmap(
    std_matrix,
    x_vals=r_stoch_vals,
    y_vals=sigma_vals,
    xlabel="r (темп роста от 0.5 до 2.0)",
    ylabel="σ (уровень шума ∈ {0.0, 0.1, 0.5})",
    title=f"Стохастическая модель (Рикер, K = {K_stoch_fixed}): Std(N_T) от r и σ",
    cmap="cividis",
    cbar_label="Std(N_T)",
    num_xticks=8,
    num_yticks=len(sigma_vals)
)

plt.show()
