import numpy as np

# --- 1. ثابت‌های فیزیکی نظریه احسان ---
PHI = (1 + np.sqrt(5)) / 2        # نسبت طلایی Φ ≈ 1.618
PHI_INV = 1 / PHI                 # 1/Φ ≈ 0.618


# --- 2. تابع آنتروپی / تنوع جمعیت ---
def calculate_entropy(pop):
    """
    محاسبه تنوع بر اساس فاصله از مرکز جرم جمعیت.
    (نسبت میانگین فاصله / فاصله نمونه‌ای)
    """
    if pop.shape[0] < 2:
        return 0

    center = np.mean(pop, axis=0)
    distances = np.linalg.norm(pop - center, axis=1)

    # برای جلوگیری از تقسیم بر صفر
    denom = np.linalg.norm(pop[0] - center)
    if denom < 1e-12:
        denom = 1.0

    return np.mean(distances) / denom



# --- 3. کلاس اصلی الگوریتم ---
class EHSANOptimizer:
    """
    نسخه کامل و نهایی EHSAN مطابق PDF (Golden Expansion Physics Algorithm)
    """

    def __init__(self, fitness_func, bounds, pop_size=50, generations=500):
        self.fitness_func = fitness_func
        self.pop_size = pop_size
        self.generations = generations

        # مدیریت محدوده‌ها
        if isinstance(bounds[0], tuple):
            self.bounds = np.array(bounds)
        else:
            raise ValueError("bounds must be list of (low, high) tuples!")

        self.dimensions = len(bounds)
        self.lower_bound = self.bounds[:, 0]
        self.upper_bound = self.bounds[:, 1]

        # حافظه نخبگان (1/Φ از جمعیت)
        self.num_elites = max(1, int(self.pop_size * PHI_INV))
        self.elite_memory = np.zeros((self.num_elites, self.dimensions))


    # --- 4. اجرای الگوریتم ---
    def run(self):
        # 4.1 جمعیت اولیه
        population = np.random.uniform(
            self.lower_bound, self.upper_bound,
            (self.pop_size, self.dimensions)
        )

        for gen in range(self.generations):

            # 4.2 ارزیابی فیتنس
            fits = np.array([self.fitness_func(p) for p in population])
            sorted_idx = np.argsort(fits)

            # بهترین کلی
            X_best = population[sorted_idx[0]]

            # به‌روزرسانی حافظه نخبگان
            self.elite_memory = population[sorted_idx[:self.num_elites]].copy()

            # --- 5. محاسبه آنتروپی جهت تشخیص حالت انفجار/انقباض ---
            R = calculate_entropy(population)

            # --- 6. سه سطح PV-Burst مطابق PDF ---
            if R < 0.01:
                # ULTRA burst → تقریباً ریست کامل
                num_reset = self.pop_size - self.num_elites
                reset_idx = sorted_idx[self.num_elites:]
                population[reset_idx] = np.random.uniform(
                    self.lower_bound, self.upper_bound,
                    (num_reset, self.dimensions)
                )

            elif R < 0.03:
                # MACRO burst → 30 تا 60 درصد ریست
                frac = np.random.uniform(0.30, 0.60)
                num_reset = int(self.pop_size * frac)
                reset_idx = sorted_idx[-num_reset:]
                population[reset_idx] = np.random.uniform(
                    self.lower_bound, self.upper_bound,
                    (num_reset, self.dimensions)
                )

            elif R < 0.10:
                # MICRO burst → فقط بدترین‌ها
                num_reset = int(self.pop_size * PHI_INV)
                reset_idx = sorted_idx[-num_reset:]
                population[reset_idx] = np.random.uniform(
                    self.lower_bound, self.upper_bound,
                    (num_reset, self.dimensions)
                )

            # --- 7. ایجاد نسل جدید با جهش طلایی ---
            new_population = []

            for i in range(self.pop_size):
                X = population[i]

                # --- 7.1 نزدیک‌ترین نخبه (Golden Refuge Local) ---
                distances = np.linalg.norm(self.elite_memory - X, axis=1)
                X_best_local = self.elite_memory[np.argmin(distances)]

                # --- 7.2 انتخاب تصادفی + بدترین جهت نیروی انتشار ---
                r = np.random.randint(0, self.pop_size)
                X_random = population[r]
                X_worst = population[sorted_idx[-1]]

                # --- 7.3 جهش برداری طلایی ---
                golden_mutation = (
                    PHI * (X_best_local - X) +
                    PHI_INV * (X_random - X_worst)
                )

                # نویز فیزیکی کنترل‌شده (خیلی مهم)
                noise = np.random.normal(0, 0.01, self.dimensions)

                X_new = X + golden_mutation + noise

                # محدودیت‌ها
                X_new = np.clip(X_new, self.lower_bound, self.upper_bound)

                new_population.append(X_new)

            # --- 8. ترکیب و انتخاب ---
            combined = np.vstack([population, np.array(new_population)])
            combined_fits = np.array([self.fitness_func(p) for p in combined])

            sorted_idx = np.argsort(combined_fits)
            population = combined[sorted_idx[:self.pop_size]]

        # --- 9. خروجی ---
        best_final = population[0]
        return best_final, float(self.fitness_func(best_final))



# --- 10. تست نمونه (Schwefel) ---
if __name__ == "__main__":

    DIM = 10

    def schwefel(p):
        return 418.9829 * DIM - np.sum(p * np.sin(np.sqrt(np.abs(p))))

    bounds = [(-500, 500)] * DIM

    print("\nRunning EHSAN on Schwefel (D=10)...\n")

    opt = EHSANOptimizer(schwefel, bounds, pop_size=50, generations=500)
    best_sol, best_fit = opt.run()

    print("Best fitness =", best_fit)
    if best_fit < 0.01:
        print(">>> SUCCESS: reached global optimum (≈0.0)")
    else:
        print(">>> NOT OPTIMAL: stuck in a local basin")

