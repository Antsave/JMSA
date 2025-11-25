# Particle Swarm Optimization (PSO)
# Simple global-best PSO for continuous optimization.


import numpy as np

def PSO(f, dim, lb, ub, num_particles=30, max_iter=500, w=0.7, c1=1.5, c2=1.5):
    particles = np.random.uniform(lb, ub, (num_particles, dim))
    velocities = np.zeros_like(particles)

    personal_best = particles.copy()
    personal_best_f = np.array([f(p) for p in particles])

    gbest_idx = np.argmin(personal_best_f)
    global_best = personal_best[gbest_idx].copy()

    curve = []

    for _ in range(max_iter):
        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()

            velocities[i] = (
                w * velocities[i] +
                c1 * r1 * (personal_best[i] - particles[i]) +
                c2 * r2 * (global_best - particles[i])
            )

            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], lb, ub)

            fit = f(particles[i])
            if fit < personal_best_f[i]:
                personal_best[i] = particles[i]
                personal_best_f[i] = fit

        gbest_idx = np.argmin(personal_best_f)
        global_best = personal_best[gbest_idx].copy()
        curve.append(f(global_best))

    return global_best, curve
