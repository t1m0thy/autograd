import numpy as np
from autograd import grad

import matplotlib.pyplot as plt

# Based on http://www.intpowertechcorp.com/GDC03.pdf

rows = 30
cols = 31
dt = 1.1
num_timesteps = 24
diff = 0.01
num_solver_iters = 5

def plot_fluid(ax, vx, vy):
    ax.clear()
    ax.matshow(np.abs(vx) +np.abs(vy))
    plt.draw()
    plt.pause(0.05)

def diffuse(x):
    """Stably diffuse by applying a few iterations of Gauss-Seidel."""
    y = x.copy()
    for k in xrange(num_solver_iters):
        y = x + diff*dt*(np.roll(y, 1, axis=0) + np.roll(y, -1, axis=0)
                       + np.roll(y, 1, axis=1) + np.roll(y, -1, axis=1))/(1 + diff*dt*4.0)
    return y

def project(u, v):
    """Project the velocity field to be mass-conserving,
       again using a few iterations of Gauss-Seidel."""
    p = np.zeros((rows, cols))
    h = 1.0/(max(rows,cols));
    div = -0.5 * h * (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)
                    + np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1))

    for k in xrange(num_solver_iters):
        p = (div + np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0)
                 + np.roll(p, 1, axis=1) + np.roll(p, -1, axis=1))/4.0

    u -= 0.5*(np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0))/h;
    v -= 0.5*(np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1))/h;
    return u, v

def advect(f, u, v):
    """Move field f according to x and y gradients (u and v)
       using an implicit Euler integrator."""
    cell_ys, cell_xs = np.mgrid[0:rows, 0:cols]
    center_xs = (cell_ys - dt * u).ravel()
    center_ys = (cell_xs - dt * v).ravel()

    # Compute indices of source cells.
    i0 = np.floor(center_xs).astype(np.int);
    j0 = np.floor(center_ys).astype(np.int);
    s1 = center_xs - i0;          # Relative weight of between two cells.
    t1 = center_ys - j0;
    i0 = np.mod(i0,     rows)     # Wrap around edges of simulation.
    i1 = np.mod(i0 + 1, rows)
    j0 = np.mod(j0,     cols)
    j1 = np.mod(j0 + 1, cols)

    # A linearly-weighted sum of the 4 surrounding cells.
    flat_f = (1 - s1) * ((1 - t1)*f[i0, j0] + t1*f[i0, j1]) \
           +       s1 * ((1 - t1)*f[i1, j0] + t1*f[i1, j1])
    return np.reshape(flat_f, (rows, cols))

if __name__ == '__main__':

    np.random.seed(1)
    #u = np.random.randn(rows, cols)
    #v = np.random.randn(rows, cols)
    u = np.ones((rows, cols)) * -0.01
    v = np.ones((rows, cols)) * -0.01
    u[10:20, 10:20] = 10.1

    fig = plt.figure(figsize=(12,10))
    ax = fig.add_axes([0., 0., 1., 1.], frameon=False)

    for t in xrange(num_timesteps):
        u = diffuse(u)
        v = diffuse(v)
        u, v = project(u, v)
        u2 = advect(u, u, v)
        v2 = advect(v, u, v)
        u, v = u2, v2
        u, v = project(u, v)
        plot_fluid(ax, u, v)

