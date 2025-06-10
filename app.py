import io
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pdfkit
import g4f

# Определение функций анализа

def analyze_behavior(time_series: np.ndarray) -> str:
    std = np.std(time_series[-int(len(time_series)/2):])
    if std < 1e-3:
        return "Стационарность"
    peaks = np.sum(np.diff(np.sign(np.diff(time_series))) < 0)
    if peaks > 5:
        return "Периодические колебания"
    return "Хаос"

def sensitivity_heatmap(model_func, param_ranges: dict, fixed_args: dict, T: int):
    p1, p2 = list(param_ranges.keys())
    v1 = np.linspace(*param_ranges[p1])
    v2 = np.linspace(*param_ranges[p2])
    amp = np.zeros((len(v1), len(v2)))
    for i, x in enumerate(v1):
        for j, y in enumerate(v2):
            args = fixed_args.copy()
            args[p1], args[p2] = x, y
            ts = model_func(*args.values(), T)
            amp[i,j] = ts.max() - ts.min()
    fig, ax = plt.subplots()
    c = ax.pcolormesh(v1, v2, amp.T, shading='auto')
    fig.colorbar(c, ax=ax)
    ax.set_xlabel(p1)
    ax.set_ylabel(p2)
    return fig

def optimize_parameters(model_func, data: np.ndarray, initial_guess: list, bounds: list, T: int):
    def loss(params):
        sim = model_func(params[0], params[1], params[2], T)
        return np.mean((sim - data)**2)
    res = minimize(loss, initial_guess, bounds=bounds)
    return res

def generate_pdf_report(model_name: str, ts: np.ndarray):
    html = f"""
    <h1>Отчёт по модели: {model_name}</h1>
    <p>Первые 10 значений:</p>
    <pre>{ts[:10]}</pre>
    <p>Размерность массива: {ts.shape}</p>
    """
    output_path = "population_report.pdf"
    pdfkit.from_string(html, output_path)
    return output_path

# Здесь далее подключаются определения simulate_* и основная логика Streamlit (не показана для краткости)
