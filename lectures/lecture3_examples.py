#!/usr/bin/ipython3

# Imports and style
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-colorblind')
plt.ion()

dx = 0.1
x = np.arange(0, 6 * np.pi, dx)
sinx = np.sin(x)
cosx = np.cos(x)

# The hard way
# fwd_diff = np.zeros(x.size - 1)
# for i in range(x.size - 1):
#     fwd_diff[i] = x[x+1] - x[i]

# The easy way
fwd_diff = (sinx[1:] - sinx[:-1]) / dx # s1 - s0 ... sn - sn-1
bkd_diff = (sinx[1:] - sinx[:-1]) / dx # s0 - sn ... sn - sn-1

plt.plot(x, cosx, label=r'Analytical Derivative of $\sin{x}$')
plt.plot(x[:-1], fwd_diff, label='Forward Difference Approx.')
plt.plot(x[1:], bkd_diff, label='Backward Difference Approx.')