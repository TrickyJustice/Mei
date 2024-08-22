import numpy as np
import torch
import math

def frange_cycle_cosine(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)  # step is in [0,1]

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and int(i + c * period) < n_epoch:
            L[int(i + c * period)] = 0.5 - 0.5 * math.cos(v * math.pi)
            v += step
            i += 1
    return L 

beta_np = frange_cycle_cosine(0, 2, 40000, 4, 0.25)
beta = torch.tensor(beta_np, dtype=torch.float32)

print(beta.shape)
