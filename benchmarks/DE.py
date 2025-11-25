# Differential Evolution (DE)
# Classic DE/rand/1/bin strategy.


import numpy as np

def DE(f, dim, lb, ub, pop_size=30, F=0.5, CR=0.9, max_iter=500):
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    curve = []

    for _ in range(max_iter):
        for i in range(pop_size):
            # mutation indices
            a, b, c = pop[np.random.choice(pop_size, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), lb, ub)

            trial = pop[i].copy()
            for j in range(dim):
                if np.random.rand() < CR:
                    trial[j] = mutant[j]

            if f(trial) < f(pop[i]):
                pop[i] = trial

        best = min([f(ind) for ind in pop])
        curve.append(best)

    best_idx = np.argmin([f(ind) for ind in pop])
    return pop[best_idx], curve
