import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import streamlit as st
except ModuleNotFoundError:
  sys.exit("Error: Streamlit is not available in this environment. Please run this script locally using 'streamlit run app.py'.")

# ==== Simulation functions ==== #

def simulate_logistic(N0, r, K, T):
    Ns = [N0]
    for _ in range(T):
        Nt = Ns[-1]
        Ns.append(Nt + r * Nt * (1 - Nt / K))
    return np.array(Ns)

def simulate_ricker(N0, r, K, T):
    Ns = [N0]
    for _ in range(T):
        Nt = Ns[-1]
        Ns.append(Nt * np.exp(r * (1 - Nt / K)))
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
        N_t = Ns[t]
        N_tau = Ns[t - tau]
        Ns.append(N_t * np.exp(r * (1 - N_tau / K)))
    return np.array(Ns)

def simulate_stochastic(base_sim, *args, sigma=0.1, repeats=100):
    results = []
    for _ in range(repeats):
        traj = base_sim(*args)
        noise = np.random.normal(0, sigma, size=traj.shape)
        results.append(traj + noise)
    return np.array(results)

# ==== Streamlit UI ==== #
st.title("Population Dynamics Simulator")

model = st.sidebar.selectbox("Select model:", [
    "Logistic Growth", "Ricker Model", "Leslie Matrix", "Delay Model", "Stochastic"
])
T = st.sidebar.number_input("Time steps (T)", min_value=1, max_value=1000, value=100)

if model in ("Logistic Growth", "Ricker Model", "Delay Model", "Stochastic"):
    N0 = st.sidebar.number_input("Initial population N0", value=10.0, step=1.0)
    r = st.sidebar.number_input("Growth rate r", value=0.1, step=0.01)
    K = st.sidebar.number_input("Carrying capacity K", value=100.0, step=1.0)

if model == "Delay Model":
    tau = st.sidebar.slider("Delay (tau)", min_value=1, max_value=10, value=1)

if model == "Leslie Matrix":
    n = st.sidebar.number_input("Number of age classes", min_value=2, max_value=10, value=3)
    fertility = []
    survival = []
    st.sidebar.write("Enter fertility coefficients:")
    for i in range(n):
        fertility.append(st.sidebar.number_input(f"f_{i}", value=0.5))
    st.sidebar.write("Enter survival probabilities:")
    for i in range(n - 1):
        survival.append(st.sidebar.number_input(f"s_{i}", value=0.8))
    N0_vec = []
    st.sidebar.write("Enter initial population per age class:")
    for i in range(n):
        N0_vec.append(st.sidebar.number_input(f"N0_{i}", value=10.0))

if model == "Stochastic":
    repeats = st.sidebar.number_input("Number of repeats", min_value=1, max_value=500, value=100)
    sigma = st.sidebar.slider("Noise sigma", min_value=0.0, max_value=1.0, value=0.1)

# ==== Run simulation ==== #
if st.sidebar.button("Simulate"):
    if model == "Logistic Growth":
        traj = simulate_logistic(N0, r, K, T)
        st.line_chart(traj)

    elif model == "Ricker Model":
        traj = simulate_ricker(N0, r, K, T)
        st.line_chart(traj)

    elif model == "Leslie Matrix":
        history = simulate_leslie(N0_vec, fertility, survival, T)
        df = pd.DataFrame(history, columns=[f"Age {i}" for i in range(len(N0_vec))])
        st.line_chart(df)
        L = np.zeros((n, n))
        L[0, :] = fertility
        for i in range(1, n):
            L[i, i-1] = survival[i-1]
        lambda_val = np.linalg.eigvals(L).max().real
        st.write(f"Dominant eigenvalue (lambda): {lambda_val:.3f}")

    elif model == "Delay Model":
        traj = simulate_delay(N0, r, K, T, tau)
        st.line_chart(traj)

    elif model == "Stochastic":
        base = simulate_ricker if st.sidebar.selectbox("Base model for stochastic:", ["Logistic", "Ricker"]) == "Ricker" else simulate_logistic
        results = simulate_stochastic(base, N0, r, K, T, sigma=sigma, repeats=repeats)
        mean_traj = results.mean(axis=0)
        st.line_chart(pd.DataFrame(results.T))
        st.line_chart(mean_traj)

    # Export data
    if model in ("Logistic Growth", "Ricker Model", "Delay Model"):
        csv = pd.DataFrame({"N": traj}).to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="trajectory.csv")
    elif model == "Leslie Matrix":
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="leslie_history.csv")
    elif model == "Stochastic":
        df_all = pd.DataFrame(results)
        csv = df_all.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="stochastic_results.csv")
