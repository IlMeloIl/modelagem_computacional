import os
import numpy as np
import pandas as pd

import theoretical
import simulator
import selfsimilar
import markov_chain
import plots

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 12345
N_CUSTOMERS = 50000

def experiment_mm1():
    print("=" * 60)
    print("Experimento 1: Sistema M/M/1")
    print("=" * 60)

    lambdas = [0.2, 0.5, 0.8, 0.9, 0.95]
    mu = 1.0
    c = 1

    rows = []

    for lam in lambdas:
        print(f"  Rodando λ={lam} ...")
        sim = simulator.simulate_mmc(lam, mu, c, N_CUSTOMERS, seed=SEED)
        theo = theoretical.mm1_theoretical(lam, mu)

        if sim is None or theo is None:
            print(f"    Sistema instável para λ={lam}")
            continue

        rows.append({
            "lambda": lam,
            "mu": mu,
            "rho_sim": sim["rho"],
            "rho_teo": theo["rho"],
            "L_sim": sim["L"],
            "L_teo": theo["L"],
            "Lq_sim": sim["Lq"],
            "Lq_teo": theo["Lq"],
            "W_sim": sim["W"],
            "W_teo": theo["W"],
            "Wq_sim": sim["Wq"],
            "Wq_teo": theo["Wq"],
            "erro_L": theoretical.relative_error(sim["L"], theo["L"]),
            "erro_Lq": theoretical.relative_error(sim["Lq"], theo["Lq"]),
            "erro_W": theoretical.relative_error(sim["W"], theo["W"]),
            "erro_Wq": theoretical.relative_error(sim["Wq"], theo["Wq"]),
            "served": sim["served"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "mm1_results.csv"), index=False)
    print("\nResultados M/M/1 salvos.\n")
    print(df.round(4).to_string(index=False))

    fixed_lambda = 0.8
    mus = [0.9, 1.0, 1.1, 1.25, 1.5]
    mu_rows = []

    print(f"\n  Sensibilidade a μ com λ fixo em {fixed_lambda} ...")
    for mu_value in mus:
        print(f"  Rodando μ={mu_value} ...")
        sim = simulator.simulate_mmc(fixed_lambda, mu_value, c, N_CUSTOMERS, seed=SEED)
        theo = theoretical.mm1_theoretical(fixed_lambda, mu_value)

        if sim is None or theo is None:
            print(f"    Sistema instável para μ={mu_value}")
            continue

        mu_rows.append({
            "lambda": fixed_lambda,
            "mu": mu_value,
            "rho_sim": sim["rho"],
            "rho_teo": theo["rho"],
            "L_sim": sim["L"],
            "L_teo": theo["L"],
            "Lq_sim": sim["Lq"],
            "Lq_teo": theo["Lq"],
            "W_sim": sim["W"],
            "W_teo": theo["W"],
            "Wq_sim": sim["Wq"],
            "Wq_teo": theo["Wq"],
            "served": sim["served"],
        })

    mu_df = pd.DataFrame(mu_rows)
    mu_df.to_csv(os.path.join(RESULTS_DIR, "mm1_mu_sensitivity.csv"), index=False)
    print("\nResultados de sensibilidade a μ salvos.\n")
    print(mu_df.round(4).to_string(index=False))

    print("Experimento M/M/1 concluído.\n")

def experiment_mm1_vs_mm2():
    print("=" * 60)
    print("Experimento 2: M/M/1 vs M/M/2")
    print("=" * 60)

    params = [
        (0.8, 1.0, 1),
        (0.8, 1.0, 2),
        (0.9, 1.0, 1),
        (0.9, 1.0, 2),
        (0.95, 1.0, 1),
        (0.95, 1.0, 2),
    ]

    rows = []
    comparison_data = []

    for lam, mu, c in params:
        print(f"  Rodando λ={lam}, μ={mu}, c={c} ...")
        sim = simulator.simulate_mmc(lam, mu, c, N_CUSTOMERS, seed=SEED)
        if sim is None:
            continue
        if c == 1:
            theo = theoretical.mm1_theoretical(lam, mu)
        else:
            theo = theoretical.mmc_theoretical(lam, mu, c)
        rho_teo = theo["rho"] if theo else None
        rows.append({
            "lambda": lam,
            "mu": mu,
            "servers": c,
            "rho_sim": sim["rho"],
            "rho_teo": rho_teo,
            "L_sim": sim["L"],
            "L_teo": theo["L"] if theo else None,
            "Lq_sim": sim["Lq"],
            "Lq_teo": theo["Lq"] if theo else None,
            "W_sim": sim["W"],
            "W_teo": theo["W"] if theo else None,
            "Wq_sim": sim["Wq"],
            "Wq_teo": theo["Wq"] if theo else None,
            "served": sim["served"],
        })
        comparison_data.append(sim)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "mmc_results.csv"), index=False)
    print("\nResultados M/M/1 vs M/M/2 salvos.\n")
    print(df.round(4).to_string(index=False))

    plots.plot_mm1_vs_mm2(comparison_data)
    print("Gráficos M/M/1 vs M/M/2 gerados.\n")

def experiment_selfsimilar():
    print("=" * 60)
    print("Experimento 3: Impacto da Autossimilaridade")
    print("=" * 60)

    base_rate = 0.8
    mu = 1.0
    c = 1
    n = 20000

    print("  Gerando interchegadas Poisson e autossimilar ...")
    poisson_ia = selfsimilar.generate_poisson_interarrivals(base_rate, n, seed=100)
    self_ia = selfsimilar.generate_ss_interarrivals(base_rate, n, phi=0.95, seed=100)

    print("  Simulando sistema com tráfego Poisson ...")
    poisson_res = simulator.simulate_mmc(
        base_rate, mu, c, n, seed=SEED,
        arrival_mode="custom", custom_interarrivals=poisson_ia
    )

    print("  Simulando sistema com tráfego autossimilar ...")
    self_res = simulator.simulate_mmc(
        base_rate, mu, c, n, seed=SEED,
        arrival_mode="custom", custom_interarrivals=self_ia
    )

    print("  Estimando parâmetro Hurst nas séries de interchegadas ...")
    poisson_H = selfsimilar.estimate_hurst_variance_time(poisson_ia[:5000])
    self_H = selfsimilar.estimate_hurst_variance_time(self_ia[:5000])

    rows = [
        {"tipo": "Poisson", "H_estimado": poisson_H,
         "L": poisson_res["L"], "Lq": poisson_res["Lq"],
         "W": poisson_res["W"], "Wq": poisson_res["Wq"],
         "served": poisson_res["served"]},
        {"tipo": "Autossimilar", "H_estimado": self_H,
         "L": self_res["L"], "Lq": self_res["Lq"],
         "W": self_res["W"], "Wq": self_res["Wq"],
         "served": self_res["served"]},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "selfsimilar_results.csv"), index=False)
    print("\nResultados autossimilaridade salvos.\n")
    print(df.to_string(index=False))

    n_windows = 5000
    self_rates = selfsimilar.generate_selfsimilar_rates(n_windows, phi=0.95, base_rate=base_rate, seed=42)
    poisson_rates = np.random.default_rng(42).poisson(base_rate, n_windows)

    plots.plot_poisson_series(poisson_rates[:200])
    plots.plot_selfsimilar_series(self_rates[:200])
    plots.plot_queue_comparison(poisson_res, self_res)
    print("Gráficos de autossimilaridade gerados.\n")

def experiment_markov():
    print("=" * 60)
    print("Experimento 4: Cadeia de Markov")
    print("=" * 60)

    P = np.array([
        [0.70, 0.25, 0.05],
        [0.20, 0.60, 0.20],
        [0.10, 0.30, 0.60],
    ])
    n_steps = 10000

    print("  Calculando distribuição estacionária teórica ...")
    pi_stationary = markov_chain.stationary_distribution(P)
    print(f"  Distribuição estacionária: {pi_stationary}")

    print("  Simulando evolução dos estados ...")
    states = markov_chain.simulate_markov_states(P, 0, n_steps, seed=42)

    simulated_freqs = np.bincount(states, minlength=3) / len(states)
    mean_times = markov_chain.mean_sojourn_times(P)

    rows = []
    for i, label in enumerate(["Baixo (0)", "Médio (1)", "Alto (2)"]):
        rows.append({
            "estado": label,
            "estacionaria_teorica": pi_stationary[i],
            "frequencia_simulada": simulated_freqs[i],
            "tempo_medio": mean_times[i],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "markov_results.csv"), index=False)
    print("\nResultados da cadeia de Markov salvos.\n")
    print(df.to_string(index=False))

    plots.plot_markov_distribution(pi_stationary, simulated_freqs)
    print("Gráficos da cadeia de Markov gerados.\n")

def main():
    print("Simulador de Sistemas Estocásticos - Trabalho A2\n")
    experiment_mm1()
    experiment_mm1_vs_mm2()
    experiment_selfsimilar()
    experiment_markov()
    print("=" * 60)
    print("Todos os experimentos concluídos!")
    print(f"Resultados em: {RESULTS_DIR}/")
    print("Gráficos em: figures/")
    print("=" * 60)

if __name__ == "__main__":
    main()
