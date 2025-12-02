# # # Jordan Burmylo-Magrann

# # # Whale Optimization Algorithm (WOA)
# # # This file is designed for notebook importing.

import numpy as np

# def WOA_test(f, dim, lb, ub, num_whales=30, max_iter=500, seed = None):
#     """
#     Whale Optimization Algorithm

#     Parameters:
#         f           : objective function
#         dim         : dimension of search space
#         lb, ub      : lower/upper bounds (scalars)
#         num_whales  : number of agents
#         max_iter    : optimization iterations

#     Returns:
#         best_pos    : best position found
#         best_curve  : list of best-so-far values over time
#     """

#     if seed is not None:
#         np.random.seed(seed)

#     whales = np.random.uniform(lb, ub, (num_whales, dim))
#     fitness = np.array([f(w) for w in whales])
#     best_idx = np.argmin(fitness)
#     best_pos = whales[best_idx].copy()
#     best_curve = []

#     for t in range(max_iter):
#         a = 2 - (2 * t / max_iter)

#         for i in range(num_whales):
#             r1, r2 = np.random.rand(), np.random.rand()
#             A = 2 * a * r1 - a
#             C = 2 * r2
#             p = np.random.rand()

#             whale = whales[i]

#             if p < 0.5:
#                 if abs(A) < 1:
#                     # Encircling prey
#                     D = abs(C * best_pos - whale)
#                     new_pos = best_pos - A * D
#                 else:
#                     # Exploration (random whale)
#                     rand_idx = np.random.randint(num_whales)
#                     rand_whale = whales[rand_idx]
#                     D = abs(C * rand_whale - whale)
#                     new_pos = rand_whale - A * D
#             else:
#                 # Spiral updating
#                 D = abs(best_pos - whale)
#                 b = 1
#                 l = (np.random.rand() * 2) - 1
#                 new_pos = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_pos

#             new_pos = np.clip(new_pos, lb, ub)
#             whales[i] = new_pos

#         fitness = np.array([f(w) for w in whales])
#         best_idx = np.argmin(fitness)
#         best_pos = whales[best_idx].copy()
#         best_curve.append(f(best_pos))

#     return best_pos, best_curve





def WOA_test(f, dim, lb, ub, num_whales=30, max_iter=500, seed=None):
    """
    Corrected Whale Optimization Algorithm (WOA)
    with proper GLOBAL BEST tracking.
    """

    if seed is not None:
        np.random.seed(seed)

    # convert bounds to arrays if they are scalars
    lb = np.array(lb) if isinstance(lb, (list, tuple, np.ndarray)) else np.full(dim, lb)
    ub = np.array(ub) if isinstance(ub, (list, tuple, np.ndarray)) else np.full(dim, ub)

    # initialize whales
    whales = np.random.uniform(lb, ub, (num_whales, dim))
    fitness = np.array([f(w) for w in whales])

    # initial global best
    best_idx = np.argmin(fitness)
    global_best_pos = whales[best_idx].copy()
    global_best_val = fitness[best_idx]

    best_curve = []

    for t in range(max_iter):
        a = 2 - (2 * t / max_iter)

        for i in range(num_whales):
            r1, r2 = np.random.rand(), np.random.rand()
            A = 2 * a * r1 - a
            C = 2 * r2
            p = np.random.rand()

            whale = whales[i]

            if p < 0.5:
                if abs(A) < 1:
                    # exploit: move toward best
                    D = abs(C * global_best_pos - whale)
                    new_pos = global_best_pos - A * D
                else:
                    # explore: pick random whale
                    rand_idx = np.random.randint(num_whales)
                    rand_whale = whales[rand_idx]
                    D = abs(C * rand_whale - whale)
                    new_pos = rand_whale - A * D
            else:
                # spiral update around global best
                D = abs(global_best_pos - whale)
                b = 1
                l = (np.random.rand() * 2) - 1
                new_pos = D * np.exp(b * l) * np.cos(2 * np.pi * l) + global_best_pos

            # clip to bounds (supports vector bounds)
            new_pos = np.clip(new_pos, lb, ub)
            whales[i] = new_pos

        # evaluate generation
        fitness = np.array([f(w) for w in whales])
        best_idx = np.argmin(fitness)
        gen_best_val = fitness[best_idx]
        gen_best_pos = whales[best_idx].copy()

        # update GLOBAL best
        if gen_best_val < global_best_val:
            global_best_val = gen_best_val
            global_best_pos = gen_best_pos.copy()

        # store best-so-far
        best_curve.append(global_best_val)

    return global_best_pos, best_curve
