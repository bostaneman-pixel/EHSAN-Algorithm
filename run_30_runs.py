import numpy as np
from ehsan_optimizer.ehsan_optimizer_v1_0 import EHSANOptimizerV1_0
from ehsan_optimizer.benchmarks import sphere, ackley, rastrigin, griewank, schwefel

def run_30(name, func, bounds):
    results = []
    for _ in range(30):
        opt = EHSANOptimizerV1_0(func, bounds)
        best_f, _ = opt.run()
        results.append(best_f)

    results = np.array(results)
    print(f"\n===== {name} =====")
    print("Mean:", np.mean(results))
    print("Std :", np.std(results))
    print("Best:", np.min(results))
    print("Worst:", np.max(results))

DIM = 10
benchmarks = [
    ("Sphere", sphere, [(-500,500)]*DIM),
    ("Ackley", ackley, [(-32,32)]*DIM),
    ("Rastrigin", rastrigin, [(-5.12,5.12)]*DIM),
    ("Griewank", griewank, [(-600,600)]*DIM),
    ("Schwefel", schwefel, [(-500,500)]*DIM),
]

for name, func, bounds in benchmarks:
    run_30(name, func, bounds)
