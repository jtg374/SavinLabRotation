#%%
import numpy as np
from numpy import pi,exp,sin,cos,tanh
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#%%
# Parameters
N = 256 # number of Neurons
tau = 10 # ms
aa=0.2;bb=1;cc=-4
phi = lambda g: aa*(1+tanh(bb*(g-cc)))
dphi = lambda g: aa*bb*(1-tanh(bb*(g-cc))**2)
b = 0 # background inhibition
A = 1 # Synaptic strength
k1 = 1 # concentration parameter for recurrent excitation
k2 = .15 # recurrent inhibition
selfConnection = True

#%%
# plot nonlinearity
x = np.linspace(-5,10)
plt.plot(x,phi(x))
#%%
# connection matrix (translation invariant)
u = np.linspace(0,2*pi,N,endpoint=False)
w = A * (exp(k1*cos(u)-1) - exp(k2*cos(u)-1))
w[0] = w[0] * selfConnection

#%%
# transient input
Iext = (1+5*cos(u-pi))*0.1
stim = lambda t: (t<30)&(t>20)
forget = lambda t: (t>450) & (t<480)

#%%
# define ODE
def mainode(t,x):
    s = x[0:N]
    r = np.zeros_like(s)
    for i in range(N):
        wi = np.roll(w,i)
        gi = np.dot(wi,x) + b + A*stim(t)*Iext[i] - forget(t)
        r[i] = phi(gi)
    ds = (r-s)/tau 
    return ds

#%%
# solve ODE
tf=200
teval = np.arange(0,tf,0.1)
s0 = np.zeros(N)
s = solve_ivp(mainode,(0,tf),s0,t_eval=teval)
#%%
plt.imshow(s.y)
plt.colorbar()

#%%
plt.plot(s.y[:,-1])    
#%%
plt.plot(s.t,s.y[int(N/2),:])    

#%%
from scipy.special import iv
import numpy as np
from numpy import pi,exp,sin,cos,tanh
import matplotlib.pyplot as plt


#%%
A1=1/exp(1);A2=1/exp(1);k1=1;k2=0.15
a0J = A1*iv(0,k1) - A2*iv(0,k2)
a1J = A1*iv(1,k1) - A2*iv(1,k2)
print(a0J*pi,a1J*pi)


#%%
a0r = 2
a1r = 1
a0g = a0r*a0J*pi
a1g = a1r*a1J*pi
r = a0r/2 + a1r*cos(u-pi)
g = a0g/2 + a1g*cos(u-pi)
fprime = dphi(g)
plt.plot(r)
plt.figure()
plt.plot(g)
plt.figure()
plt.plot(fprime)


#%%
print(np.sum(r*cos(u)*(2/N)))

#%%
