#%%
import numpy as np
from numpy import pi,exp,sin,cos,sqrt
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
#%% 
# Parameters
N = 256
dx = 2*pi/N
x = np.arange(0,2*pi,dx)
tau = 10
sigma = 0.1*pi
lu = 1
A = 5
k_critical = A**2 * lu/ (sqrt(2*pi)*8*sigma)
print('k_c: ',k_critical)
k = k_critical*1
# recurrent connection
J = (exp(-(x/sigma)**2/2) + exp(-((x-2*pi)/sigma)**2/2)  )* A/sqrt(2*pi)/sigma
#%%
def conv_circ( signal, ker ):
    '''
        signal: real 1D array
        ker: real 1D array
        signal and ker must have same shape
        https://stackoverflow.com/questions/35474078/python-1d-array-circular-convolution
    '''
    return np.real(np.fft.ifft( np.fft.fft(signal)*np.fft.fft(ker) ))
    # result = np.zeros_like(signal)
    # for ii in range(len(result)):
    #     result[ii] = np.sum(signal*np.roll(ker,ii))
    # return result

# u = np.arange(0,1000,0.5)
# g2 = exp(-((u-500)/20)**2/2)/sqrt(2*pi)/20
# g3 = conv_circ(np.roll(g2,500*2),g2)*0.5
# plt.plot(g2)
# plt.plot(g3*sqrt(2))

#%% 
# mainode
def mainode(t,U,J,I_ext):
    r = np.power(U,2)
    norm = np.sum(r)*k*dx*lu + 1
    r = r/norm
    dUdt = -U + conv_circ(J,r)*dx*lu + I_ext
    return dUdt

#%%
# initial condition 
# start from steady state
U0_A = A*( 1 + sqrt(1-k/k_critical) ) / (2*sqrt(2)*sigma*k*sqrt(2*pi))
z=pi
U0 = exp(-((x-z)/sigma)**2/4) * U0_A
kwargs = {
    'J': J,
    'I_ext': 0
}
plt.plot(x,U0)

#%%
# sanity check for fixed point
r = np.power(U0,2)
norm = np.sum(r)*k*dx*lu + 1
r = r/norm
r_norm = ( 1 + sqrt(1-k/k_critical) ) / (2*k*lu*sqrt(2*pi)*sigma)
# plt.plot(x,r/r_norm)
g = conv_circ(J,r)
plt.plot(x,g*dx*lu)
plt.plot(x,U0)

#%%
# solve
sol = solve_ivp(lambda t,x: mainode(t,x,**kwargs),(0,500),1.1*U0*np.random.normal(1,0,N),t_eval=np.linspace(0,500))
plt.imshow(sol.y)
#%%
plt.plot(x,sol.y[:,0])
plt.plot(x,sol.y[:,1])

plt.plot(x,sol.y[:,-1])
plt.plot(x,U0)

#%%
