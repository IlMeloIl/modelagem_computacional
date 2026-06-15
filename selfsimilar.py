import numpy as np

def generate_selfsimilar_rates(n_windows, phi=0.95, base_rate=0.8, seed=42):
    rng = np.random.default_rng(seed)
    x = np.zeros(n_windows)
    noise = rng.normal(0, 0.1, n_windows)
    for t in range(1, n_windows):
        x[t] = phi * x[t-1] + noise[t]
    raw = np.exp(x)
    mean_raw = np.mean(raw)
    rates = (raw / mean_raw) * base_rate
    return rates

def generate_ss_interarrivals(mean_rate, n_customers, phi=0.95, seed=42):
    rng = np.random.default_rng(seed)
    base_interarrival = 1.0 / mean_rate

    buf = n_customers + 500
    noise = rng.normal(0, 0.1, buf)
    x = np.zeros(buf)
    for t in range(1, buf):
        x[t] = phi * x[t-1] + noise[t]

    multipliers = np.exp(x)
    multipliers = multipliers / np.mean(multipliers)

    base = rng.exponential(base_interarrival, n_customers)
    interarrivals = base * multipliers[:n_customers]
    return interarrivals

def generate_poisson_interarrivals(mean_rate, n_customers, seed=42):
    rng = np.random.default_rng(seed)
    return rng.exponential(1.0 / mean_rate, n_customers)

def estimate_hurst_variance_time(series, max_pow=10):
    series = np.asarray(series, dtype=float)
    n = len(series)
    m_values = [2**j for j in range(max_pow + 1) if 2**j <= n // 2]
    variances = []

    for m in m_values:
        blocks = series[:(n // m) * m].reshape(-1, m)
        variances.append(np.var(blocks.mean(axis=1), ddof=1))

    if len(variances) < 3 or any(var <= 0 for var in variances):
        return 0.5

    slope, _ = np.polyfit(np.log2(m_values), np.log2(variances), 1)
    return 1.0 + slope / 2.0
