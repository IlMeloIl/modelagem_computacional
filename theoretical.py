import math
import numpy as np

def mm1_theoretical(lmbda, mu):
    rho = lmbda / mu
    if rho >= 1:
        return None
    L = rho / (1 - rho)
    Lq = rho**2 / (1 - rho)
    W = 1.0 / (mu - lmbda)
    Wq = rho / (mu - lmbda)
    return {
        "rho": rho,
        "L": L,
        "Lq": Lq,
        "W": W,
        "Wq": Wq,
    }

def mmc_theoretical(lmbda, mu, c):
    a = lmbda / mu
    rho = lmbda / (c * mu)
    if rho >= 1:
        return None

    sum_p0 = 0.0
    for n in range(c):
        sum_p0 += (a**n) / math.factorial(n)
    term = (a**c) / (math.factorial(c) * (1 - rho))
    p0 = 1.0 / (sum_p0 + term)

    Lq = (p0 * (a**c) * rho) / (math.factorial(c) * (1 - rho)**2)
    Wq = Lq / lmbda
    W = Wq + 1.0 / mu
    L = lmbda * W

    return {
        "rho": rho,
        "L": L,
        "Lq": Lq,
        "W": W,
        "Wq": Wq,
        "p0": p0,
    }

def relative_error(sim, theo):
    if theo == 0:
        return 0.0
    return abs(sim - theo) / abs(theo)
