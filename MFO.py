
import numpy as np


def moth_flame_optimization(obj_fn, dim,bounds, n_moths=30, n_iters=200,b=1.0,seed=None):
    ''' Moth Flame Optimization
    
        Parameters:
            obj_fn: objective function
            dim_bounds: bounds
            n_moths: number of moths
            n_iters: number of iterations
            b: spiral parameter, controls tightness of spiral
            seed: random seed
            
        Returns:
            best_pos: best position
            best_score: best fitness
            history: history of best fitness values in each iteration'''

    # rng = np.random.default_rng(seed)

    # seed
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()


    # handling bounds for obj_fn
    if isinstance(bounds[0], (list, tuple, np.ndarray)):
        lb = np.array([bnd[0] for bnd in bounds], dtype=float)
        ub = np.array([bnd[1] for bnd in bounds], dtype=float)
    else:
        lb = np.full(dim, bounds[0], dtype=float)
        ub = np.full(dim, bounds[1], dtype=float)

    # initializing moth positions in the bounds
    moths = lb + (ub - lb) * rng.random((n_moths, dim))
    # fitness function w selected obj_fn
    fitness = np.array([obj_fn(moths[i]) for i in range(n_moths)])
    # sorts index best to worst based on fitness
    idx = np.argsort(fitness)

    moths = moths[idx]
    fitness = fitness[idx]

    # best moths in index are flames
    flames = moths.copy()
    flame_fitness = fitness.copy()

    # keeping track of solutions
    best_pos = flames[0].copy()
    best_score = flame_fitness[0]
    history = [best_score]


    for it in range(1, n_iters + 1):
        # number of flames decreases in each iteration
        flame_no = round(
            n_moths - it * ((n_moths - 1) / n_iters)
        )
        flame_no = max(1, min(flame_no, n_moths))

        # moths move towards flames with spiral eq.
        for i in range(n_moths):
            # moths get matched with flames
            # pick flame with same index if moth index (i) is < number of flames
            # if moth index > i, pick last available flame (number of flames gets smaller as 
            # alg goes on, repeated flame matches)
            flame_idx = i if i < flame_no else flame_no - 1
            flame = flames[flame_idx]

            # random variable
            t = rng.uniform(-1, 1, size=dim)

            # distance between moth and flame
            distance_to_flame = np.abs(flame - moths[i])
            # spiral eq.
            moths[i] = (
                # b = spiral param
                distance_to_flame * np.exp(b * t) * np.cos(2 * np.pi * t)
                + flame
            )

        # moths stay within bounds
        moths = np.clip(moths, lb, ub)

        # recalculate moth fitness
        fitness = np.array([obj_fn(moths[i]) for i in range(n_moths)])

        # combine moths and flames
        all_positions = np.vstack([flames, moths])
        all_fitness = np.concatenate([flame_fitness, fitness])

        # sort combined flames and moths by fitness
        idx = np.argsort(all_fitness)
        all_positions = all_positions[idx]
        all_fitness = all_fitness[idx]

        # picking new flames, the best moths are now selected as flames for next iter
        flames = all_positions[:n_moths]
        flame_fitness = all_fitness[:n_moths]

        # updating best flames, or solutions if it is better than previous iteration
        if flame_fitness[0] < best_score:
            best_score = flame_fitness[0]
            best_pos = flames[0].copy()

        # save history
        history.append(best_score)

    return best_pos, best_score, history 