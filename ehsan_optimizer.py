import numpy as np

# --- 1. ثابت‌های فیزیکی ---
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI


# --- 2. آنتروپی برداری‌شده ---
def calculate_entropy(pop):
    center = np.mean(pop, axis=0)
    distances = np.linalg.norm(pop - center, axis=1)
    denom = np.linalg.norm(pop[0] - center)
    if denom < 1e-12:
        denom = 1.0
    return np.mean(distances) / denom


# --- 3. نسخه بهینه‌شده EHSAN ---
class EHSANOptimizer:

    def __init__(self, fitness_func, bounds, pop_size=50, generations=500):
        self.fitness_func = fitness_func
        self.pop_size = pop_size
        self.generations = generations

        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]

        self.num_elites = max(1, int(pop_size * PHI_INV))
        self.elite_memory = np.zeros((self.num_elites, self.dim))

    # --- 4. اجرای الگوریتم ---
    def run(self):

        # جمعیت اولیه
        pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))

        for gen in range(self.generations):

            # --- 4.1 ارزیابی ---
            fits = np.apply_along_axis(self.fitness_func, 1, pop)
            idx = np.argsort(fits)

            pop = pop[idx]
            fits = fits[idx]

            best = pop[0]
            self.elite_memory = pop[:self.num_elites].copy()

            # --- 4.2 آنتروپی ---
            R = calculate_entropy(pop)

            # --- 4.3 Burst تطبیقی ---
            if R < 0.01:
                # Ultra Burst
                pop[self.num_elites:] = np.random.uniform(
                    self.lb, self.ub, (self.pop_size - self.num_elites, self.dim)
                )
            elif R < 0.03:
                # Macro Burst
                k = np.random.randint(int(0.3*self.pop_size), int(0.6*self.pop_size))
                pop[-k:] = np.random.uniform(self.lb, self.ub, (k, self.dim))
            elif R < 0.10:
                # Micro Burst
                k = int(self.pop_size * PHI_INV)
                pop[-k:] = np.random.uniform(self.lb, self.ub, (k, self.dim))

            # --- 4.4 جهش طلایی برداری‌شده ---
            elite_idx = np.argmin(
                np.linalg.norm(self.elite_memory[:, None, :] - pop[None, :, :], axis=2),
                axis=0
            )
            X_elite = self.elite_memory[elite_idx]

            rand_idx = np.random.randint(0, self.pop_size, self.pop_size)
            X_rand = pop[rand_idx]

            X_worst = pop[-1]

            # نویز تطبیقی (بهبود مهم)
            sigma = 0.05 * np.exp(-gen / (0.3 * self.generations))
            noise = np.random.normal(0, sigma, (self.pop_size, self.dim))

            golden_mut = (
                PHI * (X_elite - pop) +
                PHI_INV * (X_rand - X_worst)
            )

            new_pop = pop + golden_mut + noise
            new_pop = np.clip(new_pop, self.lb, self.ub)

            # --- 4.5 انتخاب ---
            combined = np.vstack([pop, new_pop])
            combined_fits = np.apply_along_axis(self.fitness_func, 1, combined)

            idx = np.argsort(combined_fits)
            pop = combined[idx[:self.pop_size]]

        best = pop[0]
        return best, float(self.fitness_func(best))


# --- 5. تست روی Schwefel ---
if __name__ == "__main__":

    DIM = 10

    def schwefel(p):
        return 418.9829 * DIM - np.sum(p * np.sin(np.sqrt(np.abs(p))))

    bounds = [(-500, 500)] * DIM

    opt = EHSANOptimizer(schwefel, bounds, pop_size=50, generations=500)
    best_sol, best_fit = opt.run()

    print("Best fitness =", best_fit)
    if best_fit < 0.01:
        print(">>> SUCCESS: reached global optimum")
    else:
        print(">>> NOT OPTIMAL")
