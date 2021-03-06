from __future__ import absolute_import
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from six.moves import map

def tanh(x):
    return (1.0 - np.exp(-x))  / ( 1.0 + np.exp(-x))

d_fun = grad(tanh)  # First derivative
dd_fun = grad(d_fun) # Second derivative
ddd_fun = grad(dd_fun) # Third derivative
dddd_fun = grad(ddd_fun) # Fourth derivative
ddddd_fun = grad(dddd_fun) # Fifth derivative
dddddd_fun = grad(ddddd_fun) # Sixth derivative

x = np.linspace(-7, 7, 200)
plt.plot(x, list(map(tanh, x)),
         x, list(map(d_fun, x)),
         x, list(map(dd_fun, x)),
         x, list(map(ddd_fun, x)),
         x, list(map(dddd_fun, x)),
         x, list(map(ddddd_fun, x)),
         x, list(map(dddddd_fun, x)))

plt.axis('off')
plt.savefig("tanh.png")
plt.clf()
