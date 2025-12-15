import numpy as np


class EHSANOptimizerV1_5:
    """
    EHSAN Optimizer V1.5

    Three-layer physics-inspired optimizer (Core / Mid / Shell):
    - Core: best individuals, DE-like differential step + gradient-like direction
    - Mid: attracted toward core + mild exploration
    - Shell: strong exploration with pressure/expansion dynamics

    Usage:
        obj_func: callable(x) -> scalar
        bounds  : list of (min, max) for each dimension

        opt = EHSANOptimizerV1_5(obj_func, bounds,
                                 pop_size=50,
                                 generations=120,
                                 max_evals=8000)
        best_f, best_x = opt.run()
    """

    def __init__(self,
                 obj_func,
                 bounds,
                 pop_size=50,
                 generations=120,
                 max_evals=8000,
                 core_ratio=0.3,
                 shell_ratio=0.3):
        """
        Parameters
        ----------
        obj_func : callable
            تابع هدف: f(x) -> scalar
        bounds : list of (float, float)
            کران پایین/بالا برای هر بعد
        pop_size : int
            اندازهٔ جمعیت
        generations : int
            حداکثر تعداد نسل‌ها
        max_evals : int
            حداکثر تعداد ارزیابی تابع هدف
        core_ratio : float
            نسبت افراد core (بهترین‌ها) به کل جمعیت
        shell_ratio : float
            نسبت افراد shell (بدترین‌ها) به کل جمعیت
        """
        self.obj_func = obj_func
        self.bounds = np.array(bounds, dtype=float)
        self.dim = len(bounds)
        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]

        self.pop_size = pop_size
        self.generations = generations
        self.max_evals = max_evals
        self.core_ratio = core_ratio
        self.shell_ratio = shell_ratio

        # این‌ها در initialize تنظیم می‌شوند
        self.pop = None
        self.fits = None
        self.best_x = None
        self.best_f = None
        self.eval_count = 0

    def initialize(self):
        """ایجاد جمعیت اولیه و ارزیابی آن."""
        self.pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.fits = np.array([self.obj_func(p) for p in self.pop])
        best_idx = np.argmin(self.fits)
        self.best_x = self.pop[best_idx].copy()
        self.best_f = self.fits[best_idx]
        self.eval_count = self.pop_size

    def _layer_sizes(self):
        """محاسبهٔ اندازهٔ core / mid / shell بر اساس ratios."""
        core_size = max(2, int(self.core_ratio * self.pop_size))
        shell_size = max(2, int(self.shell_ratio * self.pop_size))
        mid_size = self.pop_size - core_size - shell_size
        if mid_size < 0:
            mid_size = 0
            shell_size = self.pop_size - core_size
        return core_size, mid_size, shell_size

    def run(self):
        """اجرای الگوریتم و بازگرداندن (best_f, best_x)."""
        self.initialize()

        core_size, mid_size, shell_size = self._layer_sizes()

        for gen in range(self.generations):
            if self.eval_count >= self.max_evals:
                break

            # زمان نرمال‌شده [0, 1]
            t = gen / max(1, (self.generations - 1))

            new_pop = self.pop.copy()
            new_fits = self.fits.copy()

            # مرتب‌سازی جمعیت
            idx_sorted = np.argsort(self.fits)
            core_idx = idx_sorted[:core_size]
            mid_idx = idx_sorted[core_size:core_size+mid_size]
            shell_idx = idx_sorted[core_size+mid_size:]

            core_pop = self.pop[core_idx]

            for i in range(self.pop_size):
                Xi = self.pop[i]

                # ---------------- Core layer ----------------
                if i in core_idx:
                    # دو فرد از core برای گام شبه-DE
                    a, b = np.random.choice(core_size, 2, replace=False)
                    Xa = core_pop[a]
                    Xb = core_pop[b]

                    # گام تفاوتی (DE-like)
                    F = 0.4 + 0.3 * t
                    de_step = F * (Xa - Xb)

                    # شبه‌گرادیان بین بهترین و بدترین core
                    best_core = core_pop[0]
                    worst_core = core_pop[-1]
                    grad_dir = (best_core - worst_core)
                    norm = np.linalg.norm(grad_dir)
                    if norm > 1e-12:
                        grad_dir = grad_dir / norm
                    grad_step = (0.2 + 0.3 * t) * grad_dir

                    # فشار به سمت best global
                    pressure = 0.5 * (self.best_x - Xi)

                    # جهش کوچک
                    mut_scale = 0.008 * (1 - t)
                    mutation = mut_scale * np.random.randn(self.dim)

                    X_new = Xi + de_step + pressure + grad_step + mutation

                # ---------------- Mid layer -----------------
                elif i in mid_idx:
                    # جذب به یک core تصادفی
                    c = np.random.choice(core_idx)
                    Xc = self.pop[c]

                    pull = (0.5 + 0.3 * t) * (Xc - Xi)

                    # انبساط خفیف بین دو نقطه تصادفی
                    r1, r2 = np.random.choice(self.pop_size, 2, replace=False)
                    Xr1, Xr2 = self.pop[r1], self.pop[r2]
                    expansion = 0.3 * (1 - t) * (Xr1 - Xr2)

                    # جهش متوسط
                    mut_scale = 0.015 * (1 - t)
                    mutation = mut_scale * np.random.randn(self.dim)

                    X_new = Xi + pull + expansion + mutation

                # ---------------- Shell layer ----------------
                else:
                    # دو نقطه تصادفی برای انبساط قوی
                    r1, r2 = np.random.choice(self.pop_size, 2, replace=False)
                    Xr1, Xr2 = self.pop[r1], self.pop[r2]

                    # فشار به سمت best (در طول زمان قوی‌تر می‌شود)
                    pressure = (0.2 + 0.5 * t) * (self.best_x - Xi)

                    # انبساط قوی در اوایل، ضعیف در اواخر
                    expansion = (0.7 * (1 - t)) * (Xr1 - Xr2)

                    # جهش بزرگ در اوایل، کوچک‌تر در اواخر
                    mut_scale = 0.04 * (1 - t)
                    mutation = mut_scale * np.random.randn(self.dim)

                    X_new = Xi + pressure + expansion + mutation

                # اعمال کران‌ها
                X_new = np.clip(X_new, self.lb, self.ub)

                # ارزیابی فرد جدید
                F_new = self.obj_func(X_new)
                self.eval_count += 1

                # پذیرش اگر بهتر باشد
                if F_new < self.fits[i]:
                    new_pop[i] = X_new
                    new_fits[i] = F_new
                    # به‌روز کردن best global
                    if F_new < self.best_f:
                        self.best_f = F_new
                        self.best_x = X_new.copy()

                if self.eval_count >= self.max_evals:
                    # اگر محدودیت ارزیابی تمام شد، حلقه را بشکن
                    break

            self.pop = new_pop
            self.fits = new_fits

            if self.eval_count >= self.max_evals:
                break

        return self.best_f, self.best_x
