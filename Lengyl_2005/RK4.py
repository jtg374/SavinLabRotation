import numpy as np
from scipy.optimize import OptimizeResult

def solver_RK4(fun, t_span,y0, t_step):
    ts = np.arange(*t_span,t_step)
    ys = np.empty((len(ts),len(y0)))
    y = np.array(y0)
    for ii,t in enumerate(ts[:-1]):
        ys[ii] = y.copy()
        k1 = t_step * fun(t,            y)
        k2 = t_step * fun(t+t_step/2,   y+k1/2)
        k3 = t_step * fun(t+t_step/2,   y+k2/2)
        k4 = t_step * fun(t+t_step,     y+k3)
        y += 1/6 * (k1 + k2*2 + k3*2 + k4)
    ys = ys.T
    return OptimizeResult(t=ts,y=ys)

