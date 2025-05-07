import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

# Streamlit import with exit if unavailable
try:
    import streamlit as st
except ModuleNotFoundError:
    sys.exit("Error: Streamlit is not available. Please install and run locally: `streamlit run app.py`.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==== Simulation functions ==== #
def simulate_logistic(N0, r, K, T):
    Ns = [N0]
    for _ in range(T):
        Ns.append(Ns[-1] + r * Ns[-1] * (1 - Ns[-1] / K))
    return np.array(Ns)

def simulate_ricker(N0, r, K, T):
    Ns = [N0]
    for _ in range(T):
        Ns.append(Ns[-1] * np.exp(r * (1 - Ns[-1] / K)))
    return np.array(Ns)

def simulate_leslie(N0_vec, fertility, survival, T):
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

def simulate_delay(N0, r, K, T, tau):
    Ns = [N0] * (tau + 1)
    for t in range(tau, T + tau):
        Ns.append(Ns[t] * np.exp(r * (1 - Ns[t - tau] / K)))
    return np.array(Ns)

def simulate_stochastic(base_sim, *args, sigma=0.1, repeats=100):
    runs = []
    for _ in range(repeats):
        traj = base_sim(*args)
        noise = np.random.normal(0, sigma, size=traj.shape)
        runs.append(traj + noise)
    return np.array(runs)

# ==== Streamlit UI ==== #
st.set_page_config(page_title="Population Dynamics Simulator", layout="wide")
st.title("🌱 Population Dynamics Simulator")

# Sidebar: model selection
st.sidebar.markdown("### Model Selection")
model = st.sidebar.selectbox("Choose model:", [
    "Logistic Growth", "Ricker Model", "Leslie Matrix", "Delay Model", "Stochastic"
])

# Sidebar: common parameters
st.sidebar.markdown("### Common Parameters")
T = st.sidebar.number_input("Time steps (T)", min_value=1, max_value=1000, value=100)

common_inputs = {"N0": None, "r": None, "K": None}
if model != "Leslie Matrix":
    common_inputs["N0"] = st.sidebar.number_input("Initial population N0", value=10.0)
    common_inputs["r"] = st.sidebar.number_input("Growth rate r", value=0.1)
    common_inputs["K"] = st.sidebar.number_input("Carrying capacity K", value=100.0)

# Sidebar: model-specific parameters
if model == "Delay Model":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Delay Parameter")
    tau = st.sidebar.slider("Delay (tau)", min_value=1, max_value=10, value=1)

if model == "Leslie Matrix":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Leslie Matrix Parameters")
    n = st.sidebar.number_input("Number of age classes", min_value=2, max_value=10, value=3)
    with st.sidebar.expander("Fertility coefficients (f_i)", expanded=False):
        fertility = [st.number_input(f"f_{i}", min_value=0.0, value=0.5) for i in range(n)]
    with st.sidebar.expander("Survival probabilities (s_i)", expanded=False):
        survival = [st.number_input(f"s_{i}", min_value=0.0, max_value=1.0, value=0.8) for i in range(n-1)]
    with st.sidebar.expander("Initial population per age class", expanded=False):
        N0_vec = [st.number_input(f"N0_{i}", min_value=0.0, value=10.0) for i in range(n)]

if model == "Stochastic":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Stochastic Parameters")
    repeats = st.sidebar.number_input("Number of repeats", min_value=1, max_value=500, value=100)
    sigma = st.sidebar.slider("Noise sigma", min_value=0.0, max_value=1.0, value=0.1)
    base_model = st.sidebar.selectbox("Base model:", ["Logistic", "Ricker"])

# Run simulation
if st.sidebar.button("Simulate"):
    with st.spinner("Simulating dynamics..."):
        if model == "Logistic Growth":
            traj = simulate_logistic(common_inputs["N0"], common_inputs["r"], common_inputs["K"], T)
            st.subheader("Logistic Growth Dynamics")
            st.line_chart(traj)

        elif model == "Ricker Model":
            traj = simulate_ricker(common_inputs["N0"], common_inputs["r"], common_inputs["K"], T)
            st.subheader("Ricker Model Dynamics")
            st.line_chart(traj)

        elif model == "Leslie Matrix":
            history = simulate_leslie(N0_vec, fertility, survival, T)
            df = pd.DataFrame(history, columns=[f"Age {i}" for i in range(len(N0_vec))])
            st.subheader("Leslie Matrix Age-Structured Dynamics")
            st.line_chart(df)
            L = np.zeros((n, n))
            L[0, :] = fertility
            for i in range(1, n):
                L[i, i-1] = survival[i-1]
            lambda_val = np.max(np.real(np.linalg.eigvals(L)))
            st.write(f"Dominant eigenvalue (λ): {lambda_val:.3f}")

        elif model == "Delay Model":
            traj = simulate_delay(common_inputs["N0"], common_inputs["r"], common_inputs["K"], T, tau)
            st.subheader("Delay Model Dynamics")
            st.line_chart(traj)

        elif model == "Stochastic":
            base_sim = simulate_ricker if base_model == "Ricker" else simulate_logistic
            results = simulate_stochastic(
                base_sim,
                common_inputs["N0"],
                common_inputs["r"],
                common_inputs["K"],
                T,
                sigma=sigma,
                repeats=repeats
            )
            st.subheader("Stochastic Simulations")
            st.line_chart(pd.DataFrame(results.T))
            st.line_chart(results.mean(axis=0))

    # Export results
    col1, col2 = st.columns(2)
    with col1:
        if model in ("Logistic Growth", "Ricker Model", "Delay Model"):
            csv = pd.DataFrame({"N": traj}).to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="trajectory.csv")
    with col2:
        if model == "Leslie Matrix":
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="leslie_history.csv")
        elif model == "Stochastic":
            df_all = pd.DataFrame(results)
            csv = df_all.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="stochastic_results.csv")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Developed by Liya Akhmetova — v1.0")
