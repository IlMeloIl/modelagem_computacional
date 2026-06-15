import os
import numpy as np
import matplotlib.pyplot as plt

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, name), dpi=150, bbox_inches="tight")
    plt.close()

def plot_mm1_vs_mm2(comparison_data):
    lambdas = [d["lambda"] for d in comparison_data if d["servers"] == 1]
    mm1 = [d for d in comparison_data if d["servers"] == 1]
    mm2 = [d for d in comparison_data if d["servers"] == 2]

    _, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = ["rho", "Lq", "Wq", "W"]
    titles = ["Utilização ρ", "Lq (fila média)", "Wq (tempo médio em fila)", "W (tempo médio no sistema)"]
    for ax, metric, title in zip(axes.flat, metrics, titles):
        ax.plot(lambdas, [d[metric] for d in mm1], "o-", label="M/M/1")
        ax.plot(lambdas, [d[metric] for d in mm2], "s--", label="M/M/2")
        ax.set_xlabel("λ")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

    save("mm1_vs_mm2.png")

def plot_poisson_series(rates):
    _, ax = plt.subplots(figsize=(12, 3))
    ax.plot(rates, alpha=0.7)
    ax.set_ylabel("Chegadas por janela")
    ax.set_xlabel("Janela de tempo")
    ax.set_title("Série temporal - Poisson")
    ax.grid(True)
    save("poisson_series.png")

def plot_selfsimilar_series(rates):
    _, ax = plt.subplots(figsize=(12, 3))
    ax.plot(rates, alpha=0.7, color="orange")
    ax.set_ylabel("Chegadas por janela")
    ax.set_xlabel("Janela de tempo")
    ax.set_title("Série temporal - Autossimilar (H > 0.5)")
    ax.grid(True)
    save("selfsimilar_series.png")

def plot_queue_comparison(poisson_res, self_res):
    labels = ["Poisson", "Autossimilar"]
    metrics = ["Lq", "Wq", "L", "W"]
    titles = ["Lq (fila média)", "Wq (tempo médio em fila)", "L (média no sistema)", "W (tempo médio no sistema)"]

    _, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, metric, title in zip(axes.flat, metrics, titles):
        vals = [poisson_res[metric], self_res[metric]]
        bars = ax.bar(labels, vals, color=["steelblue", "tomato"])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.grid(True, axis="y")

    save("poisson_vs_selfsimilar_queue.png")

def plot_markov_distribution(stationary, simulated_freqs):
    labels = ["Baixo (0)", "Médio (1)", "Alto (2)"]
    x = np.arange(len(labels))
    width = 0.35

    _, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, stationary, width, label="Estacionária (teórica)", color="steelblue")
    ax.bar(x + width / 2, simulated_freqs, width, label="Simulada", color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Probabilidade / Frequência")
    ax.set_title("Distribuição dos estados - Teórica vs Simulada")
    ax.legend()
    ax.grid(True, axis="y")
    save("markov_states_distribution.png")
