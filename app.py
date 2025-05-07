import sys
import io
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import streamlit as st
except ModuleNotFoundError:
    sys.exit("Error: Streamlit is not available. Please install and run locally: `streamlit run app.py`.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==== Simulation functions ==== #
@st.cache_data(max_entries=5)
def simulate_logistic(N0: float, r: float, K: float, T: int) -> np.ndarray:
    """Discrete logistic growth with error handling"""
    Ns = [N0]
    try:
        for _ in range(T):
            next_N = Ns[-1] + r * Ns[-1] * (1 - Ns[-1] / K)
            if next_N < 0 or np.isnan(next_N):
                return np.array(Ns)
            Ns.append(next_N)
        return np.array(Ns)
    except Exception as e:
        logger.error(f"Logistic simulation failed: {str(e)}")
        return np.array([N0])

@st.cache_data(max_entries=5)
def simulate_ricker(N0: float, r: float, K: float, T: int) -> np.ndarray:
    """Ricker model with error handling"""
    Ns = [N0]
    try:
        for _ in range(T):
            next_N = Ns[-1] * np.exp(r * (1 - Ns[-1] / K))
            if next_N < 0 or np.isnan(next_N):
                return np.array(Ns)
            Ns.append(next_N)
        return np.array(Ns)
    except Exception as e:
        logger.error(f"Ricker simulation failed: {str(e)}")
        return np.array([N0])

@st.cache_data(max_entries=5)
def simulate_leslie(N0_vec: list, fertility: list, survival: list, T: int) -> np.ndarray:
    """Leslie matrix model with validation"""
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
            N = np.clip(N, 0, None)  # Prevent negative values
            history.append(N.copy())
        return np.array(history)
    except Exception as e:
        logger.error(f"Leslie simulation failed: {str(e)}")
        return np.array([N0_vec])

@st.cache_data(max_entries=5)
def simulate_delay(N0: float, r: float, K: float, T: int, tau: int) -> np.ndarray:
    """Delay model with error handling"""
    try:
        Ns = [N0] * (tau + 1)
        for t in range(tau, T + tau):
            next_N = Ns[t] * np.exp(r * (1 - Ns[t - tau] / K))
            if next_N < 0 or np.isnan(next_N):
                return np.array(Ns)
            Ns.append(next_N)
        return np.array(Ns)
    except Exception as e:
        logger.error(f"Delay simulation failed: {str(e)}")
        return np.array([N0])

def simulate_stochastic(base_sim, *args, sigma: float = 0.1, repeats: int = 100) -> np.ndarray:
    """Stochastic simulation with progress bar"""
    runs = []
    progress = st.progress(0)
    try:
        for i in range(repeats):
            traj = base_sim(*args)
            noise = np.random.normal(0, sigma, size=traj.shape)
            runs.append(traj + noise)
            progress.progress((i + 1) / repeats)
        return np.array(runs)
    except Exception as e:
        logger.error(f"Stochastic simulation failed: {str(e)}")
        return np.array([base_sim(*args)])

# ==== Validation ==== #
def validate_leslie_params(survival: list, fertility: list):
    if sum(survival) > 1.0:
        raise ValueError("Sum of survival probabilities cannot exceed 1.0")
    if any(f < 0 for f in fertility):
        raise ValueError("Fertility coefficients cannot be negative")
    if any(not (0 <= s <= 1) for s in survival):
        raise ValueError("Survival probabilities must be in [0,1]")

# ==== Original Plotting Function ==== #
def plot_and_export(data, title):
    """Original simple plotting without log scale"""
    fig, ax = plt.subplots()
    if data.ndim == 1:
        ax.plot(data)
    else:
        ax.plot(data.T, alpha=0.1)
        ax.plot(np.mean(data, axis=0), color='red', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Population size')
    st.pyplot(fig)
    
    # Export PNG
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    st.download_button(
        "Download plot PNG", 
        data=buf, 
        file_name=f"{title.replace(' ', '_')}.png", 
        mime="image/png"
    )

# ==== Streamlit UI ==== #
st.set_page_config(page_title="Population Dynamics Simulator", layout="wide")
st.title("🌱 Population Dynamics Simulator")

# Model info
model_info = {
    "Logistic Growth": "Classic logistic map with carrying capacity K",
    "Ricker Model": "Exponential growth with density dependence",
    "Leslie Matrix": "Age-structured model via Leslie matrix",
    "Delay Model": "Population depends on past state (delay tau)",
    "Stochastic": "Adds Gaussian noise to multiple runs"
}

# Sidebar
st.sidebar.markdown("### Model Selection")
model = st.sidebar.selectbox("Choose model:", list(model_info.keys()))
st.sidebar.caption(model_info[model])

# Common parameters
st.sidebar.markdown("### Common Parameters")
T = st.sidebar.number_input("Time steps (T)", min_value=1, max_value=500, value=100)

common = {}
if model != "Leslie Matrix":
    common['N0'] = st.sidebar.number_input("Initial population N0", min_value=0.0, value=10.0)
    common['r'] = st.sidebar.number_input("Growth rate r", min_value=0.0, value=0.1)
    if common['r'] > 3.0:
        st.sidebar.warning("High growth rate (r > 3) may cause chaotic behavior!")
    common['K'] = st.sidebar.number_input("Carrying capacity K", min_value=1.0, value=100.0)

# Model-specific parameters
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

elif model == "Stochastic":
    repeats = st.sidebar.number_input("Number of repeats", min_value=1, max_value=200, value=100)
    sigma = st.sidebar.slider("Noise sigma", min_value=0.0, max_value=1.0, value=0.1)
    base_model = st.sidebar.selectbox("Base model:", ["Logistic Growth", "Ricker Model"])

# Simulation
if st.sidebar.button("Run Simulation"):
    try:
        if model == "Logistic Growth":
            traj = simulate_logistic(**common, T=T)
            st.subheader("Logistic Growth")
            plot_and_export(traj, 'Logistic Growth')

        elif model == "Ricker Model":
            traj = simulate_ricker(**common, T=T)
            st.subheader("Ricker Model")
            plot_and_export(traj, 'Ricker Model')

        elif model == "Delay Model":
            traj = simulate_delay(**common, T=T, tau=tau)
            st.subheader("Delay Model")
            plot_and_export(traj, 'Delay Model')

        elif model == "Leslie Matrix":
            history = simulate_leslie(N0_vec, fertility, survival, T)
            st.subheader("Leslie Matrix")
            plot_and_export(history, 'Leslie Matrix')
            # Dominant eigenvalue
            L = np.zeros((n, n))
            L[0, :] = fertility
            for i in range(1, n):
                L[i, i-1] = survival[i-1]
            lambda_val = np.max(np.real(np.linalg.eigvals(L)))
            st.write(f"Dominant eigenvalue λ = {lambda_val:.3f}")

        elif model == "Stochastic":
            base_sim = simulate_ricker if base_model == 'Ricker Model' else simulate_logistic
            results = simulate_stochastic(base_sim, common['N0'], common['r'], common['K'], T,
                                        sigma=sigma, repeats=repeats)
            st.subheader("Stochastic Simulation")
            plot_and_export(results, 'Stochastic Simulation')

        # Validate results
        if 'traj' in locals() and (np.isnan(traj).any() or (traj < 0).any()):
            st.warning("Simulation produced invalid values (NaN or negative)")
            
    except Exception as e:
        st.error(f"Simulation error: {str(e)}")
        logger.exception("Simulation failed")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Developed by [Your Name] — v1.0")
