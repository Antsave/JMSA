# Genetic Algorithm (GA)
# Simple continuous GA with arithmetic crossover and mutation.


import numpy as np

def GA(f, dim, lb, ub, pop_size=40, max_iter=200, mutation_rate=0.1):
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    curve = []

    for _ in range(max_iter):
        fitness = np.array([f(ind) for ind in pop])

        # Select top 50% as parents
        parents = pop[np.argsort(fitness)[:pop_size // 2]]

        children = []
        while len(children) < pop_size // 2:
            p1, p2 = parents[np.random.randint(len(parents), size=2)]
            child = (p1 + p2) / 2  # arithmetic crossover

            # mutation
            if np.random.rand() < mutation_rate:
                child += np.random.uniform(-1, 1, dim) * 0.1

            children.append(child)

        pop = np.vstack((parents, children))
        pop = np.clip(pop, lb, ub)

        best = np.min(fitness)
        curve.append(best)

    best_idx = np.argmin([f(ind) for ind in pop])
    return pop[best_idx], curve
