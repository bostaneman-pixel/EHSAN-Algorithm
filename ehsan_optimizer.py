import numpy as np

# --- 1. ثابت‌های فیزیکی بر اساس نظریه احسان ---
PHI = (1 + np.sqrt(5)) / 2       # نسبت طلایی: Φ ≈ 1.618
PHI_INV = 1 / PHI                # معکوس نسبت طلایی: 1/Φ ≈ 0.618

# --- 2. تابع محاسبه تنوع/آنتروپی (Entropy) ---
# اندازه‌گیری تنوع جمعیت برای مکانیسم پایداری PV
def calculate_entropy(population):
    """
    محاسبه تنوع جمعیت بر اساس میانگین فاصله از مرکز جرم.
    """
    if population.shape[0] < 2:
        return 0
        
    center = np.mean(population, axis=0)
    distances = np.linalg.norm(population - center, axis=1)
    
    # نرمال‌سازی با اندازه دامنه جستجو برای پایداری
    norm_factor = np.linalg.norm(population[0] - center)
    return np.mean(distances) / norm_factor if norm_factor > 1e-10 else 0

# --- 3. کلاس اصلی الگوریتم EHSAN ---
class EHSANOptimizer:
    """
    الگوریتم احسان (EHSAN): یک بهینه‌ساز سراسری الهام گرفته از فیزیک.
    از جهش برداری-طلایی تطبیقی و انفجار-انبساط چگالی برای غلبه بر بهینه‌های محلی استفاده می‌کند.
    """
    def __init__(self, fitness_func, bounds, pop_size=50, generations=500):
        """
        مقداردهی اولیه بهینه‌ساز.
        :param fitness_func: تابعی که قرار است کمینه شود (مقدار بهینه).
        :param bounds: لیست یا تاپل محدوده جستجو (مثلاً [(-500, 500)] * D).
        :param pop_size: اندازه جمعیت (تعداد کاندیداها).
        :param generations: تعداد نسل‌های اجرا.
        """
        self.fitness_func = fitness_func
        self.pop_size = pop_size
        self.generations = generations
        
        # مدیریت محدوده جستجو
        if isinstance(bounds[0], tuple):
            self.bounds = np.array(bounds)
        else:
            # اگر فقط یک محدوده کلی داده شده باشد
            self.bounds = np.array([bounds] * self.dimensions)

        self.dimensions = self.bounds.shape[0]
        self.lower_bound = self.bounds[:, 0]
        self.upper_bound = self.bounds[:, 1]
        
        # حافظه گروه چگال (1/Φ از بهترین‌ها)
        self.num_elites = int(self.pop_size * PHI_INV)
        self.elite_memory = np.empty((self.num_elites, self.dimensions)) 


    def run(self):
        """
        شروع فرایند بهینه‌سازی EHSAN.
        :return: (best_solution, best_fitness)
        """
        
        # 3.1. جمعیت اولیه تصادفی
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dimensions))
        
        for gen in range(self.generations):
            fits = np.array([self.fitness_func(p) for p in population])
            sorted_idx = np.argsort(fits)
            
            # X_best کلی (پناهگاه فشار پایین)
            X_best_global = population[sorted_idx[0]]
            
            # به‌روزرسانی حافظه گروه چگال (1/Φ بهترین‌ها)
            self.elite_memory = population[sorted_idx[:self.num_elites]].copy()
            
            # --- (الف) انفجار-انبساط چگالی (Density-Burst Expansion - پایداری PV) ---
            # اگر تنوع خیلی کم شود (فشار بالا در یک چاله محلی)
            entropy = calculate_entropy(population)
            if entropy < 0.05: # آستانه کم تنوع
                # 1/Φ از بدترین‌ها با جهش بزرگ جایگزین می‌شوند
                num_to_expand = int(self.pop_size * PHI_INV)
                worst_indices = sorted_idx[-num_to_expand:]
                
                population[worst_indices] = np.random.uniform(
                    self.lower_bound, self.upper_bound, (num_to_expand, self.dimensions)
                )

            # --- (ب) انبساط با جهش برداری-طلایی تطبیقی (Adaptive Golden-Vector Mutation) ---
            new_population = []
            for i in range(self.pop_size):
                X = population[i]
                
                # یافتن X_best_local (نزدیک‌ترین و بهترین پناهگاه محلی در حافظه)
                distances_to_elites = np.linalg.norm(self.elite_memory - X, axis=1)
                closest_elite_idx = np.argmin(distances_to_elites)
                X_best_local = self.elite_memory[closest_elite_idx]
                
                # انتخاب دو راه‌حل برای نیروی انبساط ذاتی
                r1 = np.random.choice(self.pop_size)
                while r1 == i: # مطمئن می‌شویم تصادفی نیست
                     r1 = np.random.choice(self.pop_size)
                     
                X_random = population[r1]
                X_worst = population[sorted_idx[-1]] # بدترین برای بردار قوی‌تر
                
                # فرمول جهش برداری-طلایی تطبیقی
                vector_mutation = (
                    PHI * (X_best_local - X) +      # نیروی پناهندگی طلایی محلی (Φ)
                    PHI_INV * (X_random - X_worst)  # نیروی انبساط ذاتی/رانش (1/Φ)
                )
                
                X_new = X + vector_mutation
                
                # اعمال محدودیت‌های مرزی (Boundary Check)
                X_new = np.clip(X_new, self.lower_bound, self.upper_bound)
                new_population.append(X_new)

            # --- (ج) کاهش فشار/پناهندگی (Pressure Reduction Phase) ---
            combined = np.vstack([population, np.array(new_population)])
            combined_fits = np.array([self.fitness_func(p) for p in combined])
            
            # مرتب‌سازی بر اساس فیتنس (کاهش P)
            sorted_idx = np.argsort(combined_fits)
            
            # پناهندگی (Refuge): انتخاب بهترین‌ها برای حفظ اندازه ثابت
            population = combined[sorted_idx[:self.pop_size]]

        # 4. خروجی نهایی
        best_final = population[0]
        return best_final, self.fitness_func(best_final)

# --- اجرای نمونه برای تست کد ---
if __name__ == '__main__':
    
    # تعریف تابع فیتنس نمونه (تابع Schwefel: هدف 0)
    DIMENSIONS = 10
    def schwefel_function(p):
        # Schwefel function with standard bounds
        return 418.9829 * DIMENSIONS - np.sum(p * np.sin(np.sqrt(np.abs(p))))

    # محدوده Schwefel استاندارد
    bounds_schwefel = [(-500, 500)] * DIMENSIONS
    
    # ایجاد و اجرای بهینه‌ساز EHSAN
    optimizer = EHSANOptimizer(
        fitness_func=schwefel_function,
        bounds=bounds_schwefel,
        pop_size=50,
        generations=500
    )
    
    print("--- شروع بهینه‌سازی EHSAN در تابع Schwefel (D=10) ---")
    best_solution, best_fitness = optimizer.run()
    
    print(f"\nبهترین فیتنس یافت شده (Schwefel): {best_fitness:.8f}")
    # انتظار می‌رود بهترین فیتنس بسیار نزدیک به 0.0 باشد.
    if best_fitness < 0.01:
         print("✅ تست موفق: الگوریتم EHSAN به بهینه جهانی رسید.")
    else:
         print("❌ تست ناموفق: الگوریتم در بهینه محلی گیر کرد.")
