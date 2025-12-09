# # # Jordan Burmylo-Magrann

# # # Whale Optimization Algorithm (WOA)
# # # This file is designed for notebook importing.

import numpy as np

def WOA(f, dim, lb, ub, num_whales=30, max_iter=500, seed = None):
    """
    Whale Optimization Algorithm

    Parameters:
        f           : objective function
        dim         : dimension of search space
        lb, ub      : lower/upper bounds (scalars)
        num_whales  : number of agents
        max_iter    : optimization iterations

    Returns:
        best_pos    : best position found
        best_curve  : list of best-so-far values over time
    """

    # seed
    if seed is not None:
        np.random.seed(seed)

    # bounds, whales initialized uniformly between lb and ub
    whales = np.random.uniform(lb, ub, (num_whales, dim))
    # fitness function
    fitness = np.array([f(w) for w in whales])
    # lowest fitness
    best_idx = np.argmin(fitness)
    # current global best
    best_pos = whales[best_idx].copy()
    # stores best valyes
    best_curve = []

    for t in range(max_iter):
        # a decreases from 2 to 0 linearly as t increases
        a = 2 - (2 * t / max_iter)

        for i in range(num_whales):
            # random values
            r1, r2 = np.random.rand(), np.random.rand()
            # controls how much whales move relative to others
            A = 2 * a * r1 - a
            # weight distance to random whale
            C = 2 * r2
            # random value that decides strategies
            p = np.random.rand()

            whale = whales[i]

            # deciding strategies - encircling or exploration
            if p < 0.5:
                if abs(A) < 1:
                    # Encircling prey
                    # D = weighted distance vector, moves towards best pos
                    D = abs(C * best_pos - whale)
                    # move towards other whale
                    new_pos = best_pos - A * D
                else:
                    # Exploration (random whale)
                    # picks random whale and moves closeer to it
                    rand_idx = np.random.randint(num_whales)
                    rand_whale = whales[rand_idx]
                    D = abs(C * rand_whale - whale)
                    new_pos = rand_whale - A * D
            else:
                # Spiral updating
                # distance to best whale
                D = abs(best_pos - whale)
                b = 1
                # random
                l = (np.random.rand() * 2) - 1
                # spiral formula that moves whale around best 
                new_pos = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_pos

            # make sure everything is within bounds
            new_pos = np.clip(new_pos, lb, ub)
            # updates
            whales[i] = new_pos

        # redo fitness for whales
        fitness = np.array([f(w) for w in whales])
        # redo best positions
        best_idx = np.argmin(fitness)
        best_pos = whales[best_idx].copy()
        # fix best curve
        best_curve.append(f(best_pos))

    return best_pos, best_curve


