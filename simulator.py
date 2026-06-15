import simpy
import numpy as np

def simulate_mmc(arrival_rate, service_rate, servers, n_customers, seed=12345, arrival_mode="poisson", custom_interarrivals=None):
    rng = np.random.default_rng(seed)

    env = simpy.Environment()
    resource = simpy.Resource(env, capacity=servers)

    wait_times = []
    system_times = []
    service_times_list = []

    n_served = 0

    def customer(env, name, resource, service_rate, rng):
        nonlocal n_served
        arrival = env.now
        with resource.request() as req:
            yield req
            start = env.now
            service_time = rng.exponential(1.0 / service_rate)
            yield env.timeout(service_time)
        departure = env.now
        wait_times.append(start - arrival)
        system_times.append(departure - arrival)
        service_times_list.append(service_time)
        n_served += 1

    def arrival_process(env, resource, arrival_rate, service_rate, n_customers, rng, arrival_mode, custom_interarrivals):
        if arrival_mode == "poisson":
            for i in range(n_customers):
                interarrival = rng.exponential(1.0 / arrival_rate)
                yield env.timeout(interarrival)
                env.process(customer(env, f"Customer-{i}", resource, service_rate, rng))
        elif arrival_mode == "custom" and custom_interarrivals is not None:
            for i, interarrival in enumerate(custom_interarrivals):
                if interarrival < 0:
                    interarrival = 0
                yield env.timeout(interarrival)
                env.process(customer(env, f"Customer-{i}", resource, service_rate, rng))

    env.process(arrival_process(env, resource, arrival_rate, service_rate, n_customers, rng, arrival_mode, custom_interarrivals))
    env.run()

    if n_served == 0:
        return None

    wait_times = np.array(wait_times)
    system_times = np.array(system_times)
    service_times_list = np.array(service_times_list)

    total_service = np.sum(service_times_list)
    total_time = env.now

    rho = total_service / (servers * total_time) if total_time > 0 else 0
    W = np.mean(system_times) if len(system_times) > 0 else 0
    Wq = np.mean(wait_times) if len(wait_times) > 0 else 0
    L = n_served / total_time * W if total_time > 0 else 0
    Lq = n_served / total_time * Wq if total_time > 0 else 0

    return {
        "lambda": arrival_rate,
        "mu": service_rate,
        "servers": servers,
        "rho": rho,
        "L": L,
        "Lq": Lq,
        "W": W,
        "Wq": Wq,
        "served": n_served,
        "simulation_time": total_time,
    }
