import sys
import io
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Streamlit import with exit if unavailable
try:
    import streamlit as st
except ModuleNotFoundError:
    sys.exit("Error: Streamlit is not available. Please install and run locally: `streamlit run app.py`.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==== Simulation functions ==== #
def simulate_logistic(N0: float, r: float, K: float, T: int) -> np.ndarray:
    """
    Discrete logistic growth:
    N_{t+1} = N_t + r * N_t * (1 - N_t / K)
    """
    Ns = [N0]
    for _ in range(T):
        Ns.append(Ns[-1] + r * Ns[-1] * (1 - Ns[-1] / K))
    return np.array(Ns)

def simulate_ricker(N0: float, r: float, K: float, T: int) -> np.ndarray:
    """
    Ricker model:
    N_{t+1} = N_t * exp(r * (1 - N_t / K))
    """
    Ns = [N0]
    for _ in range(T):
        Ns.append(Ns[-1] * np.exp(r * (1 - Ns[-1] / K)))
    return np.array(Ns)

def simulate_leslie(N0_vec: list, fertility: list, survival: list, T: int) -> np.ndarray:
    """
    Leslie matrix age-structured model.
    N_{t+1} = L @ N_t
    """
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
    """
    Delay model: reproduction depends on N_{t - tau}.
    """
    Ns = [N0] * (tau + 1)
    for t in range(tau, T + tau):
        Ns.append(Ns[t] * np.exp(r * (1 - Ns[t - tau] / K)))
    return np.array(Ns)

def simulate_stochastic(base_sim, *args, sigma: float = 0.1, repeats: int = 100) -> np.ndarray:
    """
    Add Gaussian noise to multiple runs of base_sim.
    """
    runs = []
    progress = st.progress(0)
    for i in range(repeats):
        traj = base_sim(*args)
        noise = np.random.normal(0, sigma, size=traj.shape)
        runs.append(traj + noise)
        progress.progress((i + 1) / repeats)
    return np.array(runs)

# ==== Streamlit UI ==== #
st.set_page_config(page_title="Population Dynamics Simulator", layout="wide")
st.title("🌱 Population Dynamics Simulator")

# Model descriptions
model_info = {
    "Logistic Growth": "Classic logistic map with carrying capacity K.",
    "Ricker Model": "Exponential growth with density dependence (Ricker).",
    "Leslie Matrix": "Age-structured model via Leslie matrix.",
    "Delay Model": "Population depends on past state (delay tau).",
    "Stochastic": "Adds Gaussian noise to multiple runs."
}
st.sidebar.markdown("### **Select a model and set parameters below.**")

# Sidebar: model selection
st.sidebar.markdown("### Model Selection")
model = st.sidebar.selectbox("Choose model:", list(model_info.keys()))
st.sidebar.caption(model_info[model])

# Sidebar: common parameters
st.sidebar.markdown("### Common Parameters")
T = st.sidebar.number_input("Time steps (T)", min_value=1, max_value=500, value=100)
if T > 500:
    st.sidebar.error("T is too large; please select ≤500")

common = {}
if model != "Leslie Matrix":
    common['N0'] = st.sidebar.number_input("Initial population N0", min_value=0.0, value=10.0)
    common['r'] = st.sidebar.number_input("Growth rate r", min_value=0.0, value=0.1)
    common['K'] = st.sidebar.number_input("Carrying capacity K", min_value=1.0, value=100.0)

# Sidebar: model-specific parameters
if model == "Delay Model":
    tau = st.sidebar.slider("Delay (tau)", min_value=1, max_value=10, value=1)

elif model == "Leslie Matrix":
    n = st.sidebar.number_input("Number of age classes", min_value=2, max_value=10, value=3)
    with st.sidebar.expander("Fertility coefficients (f_i)"):
        fertility = [st.number_input(f"f_{i}", min_value=0.0, value=0.5) for i in range(n)]
    with st.sidebar.expander("Survival probabilities (s_i)"):
        survival = [st.number_input(f"s_{i}", min_value=0.0, max_value=1.0, value=0.8) for i in range(n-1)]
    with st.sidebar.expander("Initial population per age class"):
        N0_vec = [st.number_input(f"N0_{i}", min_value=0.0, value=10.0) for i in range(n)]
    # Validate Leslie params
    if any(f < 0 for f in fertility):
        st.sidebar.error("All fertility coefficients must be ≥ 0")
    if any(not (0 <= s <= 1) for s in survival):
        st.sidebar.error("All survival probabilities must be in [0,1]")

elif model == "Stochastic":
    repeats = st.sidebar.number_input("Number of repeats", min_value=1, max_value=200, value=100)
    if repeats > 200:
        st.sidebar.error("Repeats too large; max 200")
    sigma = st.sidebar.slider("Noise sigma", min_value=0.0, max_value=1.0, value=0.1)
    base_model = st.sidebar.selectbox("Base model:", ["Logistic", "Ricker"])

# Utility: plot + export PNG
def plot_and_export(data, title):
    fig, ax = plt.subplots()
    if data.ndim == 1:
        ax.plot(data)
    else:
        ax.plot(data)
    ax.set_title(title)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Population size')
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    st.download_button("Download plot PNG", data=buf, file_name=f"{title}.png", mime="image/png")

# Simulate
if st.sidebar.button("Simulate"):
    with st.spinner("Simulating..."):
        if model == "Logistic Growth":
            traj = simulate_logistic(common['N0'], common['r'], common['K'], T)
            st.subheader("Logistic Growth")
            st.line_chart(traj)
            plot_and_export(traj, 'logistic_growth')

        elif model == "Ricker Model":
            traj = simulate_ricker(common['N0'], common['r'], common['K'], T)
            st.subheader("Ricker Model")
            st.line_chart(traj)
            plot_and_export(traj, 'ricker_model')

        elif model == "Delay Model":
            traj = simulate_delay(common['N0'], common['r'], common['K'], T, tau)
            st.subheader("Delay Model")
            st.line_chart(traj)
            plot_and_export(traj, 'delay_model')

        elif model == "Leslie Matrix":
            history = simulate_leslie(N0_vec, fertility, survival, T)
            df = pd.DataFrame(history, columns=[f"Age {i}" for i in range(n)])
            st.subheader("Leslie Matrix")
            st.line_chart(df)
            plot_and_export(df.values, 'leslie_matrix')
            # Dominant eigenvalue
            L = np.zeros((n, n)); L[0, :] = fertility
            for i in range(1, n): L[i, i-1] = survival[i-1]
            lambda_val = np.max(np.real(np.linalg.eigvals(L)))
            st.write(f"Dominant eigenvalue λ = {lambda_val:.3f}")
            st.download_button("Download data CSV", data=df.to_csv(index=False).encode('utf-8'),
                               file_name='leslie_matrix.csv')

        elif model == "Stochastic":
            base_sim = simulate_ricker if base_model == 'Ricker' else simulate_logistic
            results = simulate_stochastic(base_sim, common['N0'], common['r'], common['K'], T,
                                          sigma=sigma, repeats=repeats)
            st.subheader("Stochastic Simulation")
            st.line_chart(pd.DataFrame(results.T))
            st.write("Mean trajectory:")
            mean_traj = results.mean(axis=0)
            st.line_chart(mean_traj)
            plot_and_export(mean_traj, 'stochastic_mean')

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Developed by Liya Akhmetova — v1.0")
st.sidebar.write("Repo: https://github.com/yourrepo/population-simulator")
