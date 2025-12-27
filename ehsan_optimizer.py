import numpy as np

# ============================
# Benchmark functions
# ============================

def sphere(x): 
    return np.sum(np.asarray(x, dtype=np.float64)**2)

def rastrigin(x):
    x = np.asarray(x, dtype=np.float64); A = 10.0
    return A*len(x) + np.sum(x**2 - A*np.cos(2*np.pi*x))

def ackley(x):
    x = np.asarray(x, dtype=np.float64); a=20.0; b=0.2; c=2*np.pi; d=len(x)
    s1 = np.sum(x**2); s2 = np.sum(np.cos(c*x))
    return -a*np.exp(-b*np.sqrt(s1/d)) - np.exp(s2/d) + a + np.e

def griewank(x):
    x = np.asarray(x, dtype=np.float64); i = np.arange(1,len(x)+1, dtype=np.float64)
    return np.sum(x**2)/4000.0 - np.prod(np.cos(x/np.sqrt(i))) + 1.0

def schwefel(x):
    x = np.asarray(x, dtype=np.float64)
    return 418.9829*len(x) - np.sum(x*np.sin(np.sqrt(np.abs(x))))

# ============================
# Helpers
# ============================

def clamp(x, low, high): 
    return np.clip(x, low, high)

def random_in_bounds(dim, low, high): 
    return np.random.uniform(low, high, size=dim).astype(np.float64)

def population_density(xs, low, high):
    Xn = (xs - low) / (high - low + 1e-12)
    n, d = Xn.shape; mnn = 0.0
    for i in range(n):
        di = np.sqrt(np.sum((Xn[i] - Xn)**2, axis=1)); di[i] = np.inf
        mnn += np.min(di)
    mnn /= n; d0 = 0.5 / np.sqrt(d)
    return 1.0 - np.clip(mnn / (d0 + 1e-12), 0.0, 1.0)

def choose_triplet(xs):
    n = len(xs)
    i = np.random.randint(0, n)
    j = np.random.randint(0, n)
    k = np.random.randint(0, n)
    return i, j, k

def stable_softmax(logits):
    m = np.max(logits)
    exps = np.exp(logits - m)
    return exps / (np.sum(exps) + 1e-12)

def local_energy_variance(func, x, low, high, time_local, samples=10):
    d = len(x); span = (high - low)
    xs = x + np.random.randn(samples, d) * (0.06 * time_local * span)
    xs = clamp(xs, low, high)
    fs = np.array([func(xx) for xx in xs])
    return np.var(fs)

# ============================
# Fusion V4.2 (stable softmax + ensemble-neutral + adaptive microsearch)
# ============================

def fuse_three(xA,fA,xB,fB,xC,fC,time,low,high,func,D):
    T = max(time, 1e-12)

    # Neutral global soft base (full-fitness)
    wA = np.exp(-fA/T); wB = np.exp(-fB/T); wC = np.exp(-fC/T)
    x_base = (wA*xA + wB*xB + wC*xC) / (wA + wB + wC + 1e-12)
    x_base = clamp(x_base, low, high)
    f_base = func(x_base)

    # Per-dimension ensemble-neutral soft fusion with stable softmax on delta-energy
    d = len(xA)
    x_new = x_base.copy()
    for i in range(d):
        candidates = np.array([xA[i], xB[i], xC[i]], dtype=np.float64)
        logits = []
        for val in candidates:
            fvals = []
            for base in (x_base, xA, xB, xC):
                tmp = base.copy(); tmp[i] = val
                fvals.append(func(clamp(tmp, low, high)))
            f_mean = np.mean(fvals)
            delta = np.clip(f_mean - f_base, -1e3, 1e3)  # negative is better
            logits.append(-delta / T)
        weights = stable_softmax(np.array(logits, dtype=np.float64))
        x_soft = np.sum(weights * candidates)

        tmp_adpt = x_new.copy(); tmp_adpt[i] = x_soft
        f_adpt = func(clamp(tmp_adpt, low, high))
        if f_adpt < func(x_new):
            x_new[i] = x_soft

    # Adaptive 5-point micro line search (use eps[i])
    span = (high - low)
    var_loc = local_energy_variance(func, x_new, low, high, time_local=T, samples=10)
    eps_base = 0.05 * (1.0 - D) * np.sqrt(T)
    eps = np.clip(eps_base * (1.0 + np.sqrt(var_loc + 1e-18)), 1e-12, 0.25) * span
    for i in range(d):
        xi = x_new[i]
        cand = [xi, xi - eps[i], xi + eps[i], xi - 2*eps[i], xi + 2*eps[i]]
        fvals = []
        for v in cand:
            tmp = x_new.copy(); tmp[i] = v
            fvals.append(func(clamp(tmp, low, high)))
        x_new[i] = cand[int(np.argmin(fvals))]

    # Minimal nonlinear noise (self-regulating)
    noise_scale = 0.05 * np.sqrt(T) * (1.0 - D)**2
    x_new += noise_scale*np.random.randn(*x_new.shape)

    return clamp(x_new, low, high)

# ============================
# Optimizer V4.2
# ============================

def optimize_v42(func,dim,low,high,
                 max_rounds=500,triplets_per_round=14,seed=None,
                 time0=0.28,
                 patience=30, tol=1e-10, max_pop=30,
                 start_pop=5):
    if seed is not None: 
        np.random.seed(seed)

    low_arr  = np.full(dim, low, dtype=np.float64)
    high_arr = np.full(dim, high, dtype=np.float64)

    xs = np.array([random_in_bounds(dim,low,high) for _ in range(start_pop)], dtype=np.float64)
    fs = np.array([func(x) for x in xs], dtype=np.float64)
    best_idx = np.argmin(fs); best_x, best_f = xs[best_idx].copy(), fs[best_idx]

    time = time0; no_improve = 0

    for _ in range(max_rounds):
        D = population_density(xs, low_arr, high_arr)
        idx = np.argsort(fs); xs, fs = xs[idx], fs[idx]
        new_xs, new_fs = [], []

        for _k in range(triplets_per_round):
            i, j, k = choose_triplet(xs)
            xA, fA = xs[i], fs[i]; xB, fB = xs[j], fs[j]; xC, fC = xs[k], fs[k]
            xN = fuse_three(xA,fA,xB,fB,xC,fC,time,low_arr,high_arr,func,D)
            fN = func(xN)
            new_xs.append(xN); new_fs.append(fN)
            if fN < best_f - tol:
                best_f, best_x = fN, xN.copy(); no_improve = 0
            else:
                no_improve += 1

        keep = min(len(xs)//2, triplets_per_round)
        xs = np.vstack([xs[:keep], np.array(new_xs, dtype=np.float64)])
        fs = np.concatenate([fs[:keep], np.array(new_fs, dtype=np.float64)])

        time *= (1.0 - D**2)

        if no_improve >= patience and len(xs) < max_pop:
            perturb = 0.1*(high_arr-low_arr)*np.random.randn(dim)
            new_start = clamp(best_x+perturb, low_arr, high_arr)
            xs = np.vstack([xs, new_start]); fs = np.concatenate([fs, [func(new_start)]])
            border_point = np.where(np.random.rand(dim) < 0.5, low_arr, high_arr).astype(np.float64)
            border_point += 0.05*(high_arr-low_arr)*np.random.randn(dim)
            border_point = clamp(border_point, low_arr, high_arr)
            xs = np.vstack([xs, border_point]); fs = np.concatenate([fs, [func(border_point)]])
            no_improve = 0

        if no_improve >= patience and len(xs) >= max_pop:
            break

    return best_f, best_x

# ============================
# Example run
# ============================

if __name__ == "__main__":
    tests=[("Sphere",sphere,-5.12,5.12),
           ("Rastrigin",rastrigin,-5.12,5.12),
           ("Ackley",ackley,-32.768,32.768),
           ("Griewank",griewank,-600,600),
           ("Schwefel",schwefel,-500,500)]
    for name,f,low,high in tests:
        print("\n################################")
        print("Testing:", name)
        print("################################")
        bf,bx = optimize_v42(f,dim=10,low=low,high=high,
                             max_rounds=500,triplets_per_round=14,seed=0,
                             time0=0.28,
                             patience=30, tol=1e-10, max_pop=30,
                             start_pop=5)
        print("FINAL BEST =", bf)
        print("Best x =", bx)
