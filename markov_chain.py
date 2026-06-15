import numpy as np

def stationary_distribution(P):
    n = P.shape[0]
    A = P.T - np.eye(n)
    A = np.vstack([A, np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    return pi

def simulate_markov_states(P, initial_state, n_steps, seed=42):
    rng = np.random.default_rng(seed)
    states = np.zeros(n_steps + 1, dtype=int)
    states[0] = initial_state
    current = initial_state
    for t in range(1, n_steps + 1):
        current = rng.choice(len(P), p=P[current])
        states[t] = current
    return states

def mean_sojourn_times(P):
    n = P.shape[0]
    mean_times = []
    for i in range(n):
        mean_times.append(1.0 / (1.0 - P[i, i]))
    return np.array(mean_times)
