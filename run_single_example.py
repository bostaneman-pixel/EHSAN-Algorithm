from ehsan_optimizer.ehsan_optimizer_v1_0 import EHSANOptimizerV1_0
from ehsan_optimizer.benchmarks import sphere

DIM = 10
bounds = [(-500,500)] * DIM

opt = EHSANOptimizerV1_0(sphere, bounds)
best_f, best_x = opt.run()

print("Best value:", best_f)
print("Best point:", best_x)
