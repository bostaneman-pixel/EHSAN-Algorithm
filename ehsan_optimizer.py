import numpy as np

class EHSANOptimizerV1_0:
    def __init__(self, obj_func, bounds,
                 pop_size=80,
                 generations=300,
                 max_evals=20000):
        self.obj_func = obj_func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]

        self.pop_size = pop_size
        self.generations = generations
        self.max_evals = max_evals

    def initialize(self):
        self.pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.fits = np.array([self.obj_func(p) for p in self.pop])
        self.best_x = self.pop[np.argmin(self.fits)].copy()
        self.best_f = np.min(self.fits)
        self.eval_count = self.pop_size

    def run(self):
        self.initialize()

        for gen in range(self.generations):
            if self.eval_count >= self.max_evals:
                break

            t = gen / max(1, (self.generations - 1))

            new_pop = self.pop.copy()
            new_fits = self.fits.copy()

            for i in range(self.pop_size):
                Xi = self.pop[i]

                C1 = 0.2 * np.random.rand()
                C2 = 0.5 * (1 - t) * np.random.rand()
                C3 = (0.05 + 0.15 * t) * np.random.rand()

                Exploitation = C1 * (self.best_x - Xi)

                r1, r2 = np.random.choice(self.pop_size, 2, replace=False)
                Xr1, Xr2 = self.pop[r1], self.pop[r2]
                Exploration = C2 * (Xr1 - Xr2)

                Targeting = C3 * (self.best_x - Xi)

                if t < 0.5:
                    mutation_scale = 0.02
                elif t < 0.7:
                    mutation_scale = 0.01
                else:
                    mutation_scale = 0.0

                Mutation = mutation_scale * np.random.randn(self.dim)

                X_new = Xi + Exploitation + Exploration + Targeting + Mutation
                X_new = np.clip(X_new, self.lb, self.ub)

                F_new = self.obj_func(X_new)
                self.eval_count += 1

                if F_new < self.fits[i]:
                    new_pop[i] = X_new
                    new_fits[i] = F_new
                    if F_new < self.best_f:
                        self.best_f = F_new
                        self.best_x = X_new.copy()

            self.pop = new_pop
            self.fits = new_fits

        return self.best_f, self.best_x
