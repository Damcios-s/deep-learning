import numpy as np
import time

class Relu:
  @staticmethod
  def forward(x):
    t0 = time.time()
    out = np.maximum(0, x)
    elapsed = time.time() - t0
    return out, {'time': elapsed}

  @staticmethod
  def backward(d_out, x):
    t0 = time.time()
    out = d_out * (x > 0)
    elapsed = time.time() - t0
    return out, {'time': elapsed}
