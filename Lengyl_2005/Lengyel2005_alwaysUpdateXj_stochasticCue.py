#%%
import numpy as np
from numpy import pi,exp,sin,cos
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
#%%
# Global parameters
N = 200 # number of neurons
M = 10 # number of memorys
k_prior = 0.5 # concentration parameter for prior distribution
k_cue0 = 16 # for initial cue distribution
v_noise = 1/8 # for cue noise accumulation, k_cue(t) = 1/( 1/k_cue0 + v_noise*t/T_theta )
#%%
# Random cue generator
class storedNoise:
    decimal = 5 # decimal to round t
    def __init__(self,t=0.,k_cue0=k_cue0,v_noise=v_noise,T_theta=T_theta,N=N):
        self.N=N
        self.k_cue0 = k_cue0
        self.v_noise = v_noise
        self.T_theta = T_theta
        self._x_noise = {}
        t_store = str(np.round(t,self.decimal))
        self._x_noise[t_store] = np.random.vonmises(0,self.k_cue(t),self.N)
    def k_cue(self,t):
        return 1/( 1/self.k_cue0 + self.v_noise*t/self.T_theta )
    def get(self,t):
        t_store = str(np.round(t,self.decimal))
        stored = self._x_noise.get(t_store)
        if stored is not None:
            return stored
        else:
            self._x_noise[t_store] = np.random.vonmises(0,self.k_cue(t),self.N)
            return self.get(t)
# x_noise = storedNoise()
# print(x_noise.get(0)[:10])
# print(x_noise.get(0.000001)[:10])
# print(np.var(x_noise.get(0)))
# print(x_noise.get(1000.000002)[:10])
# print(x_noise.get(1000.000001)[:10])
# print(x_noise.get(1000.0001)[:10])
# print(np.var(x_noise.get(1000)))
#%%
## Define STDP and Phase coupling function
A_STDP = 0.03
s_STDP = 4
T_theta = 125 # theta oscillation period in ms
dp = lambda dt: dt*2*pi/T_theta # dt = xi - xj
omega = lambda dx: A_STDP * exp(s_STDP*cos(dx)) * sin(dx)
# derivative in respect to xi
domega = lambda dx: 2*pi/T_theta*A_STDP*exp(s_STDP*cos(dx)) * (cos(dx) - s_STDP * sin(dx)**2 )

#%%
## Create Memorys
xMemory = np.random.vonmises(0,k_prior,(N,M))
## Create Synapses
W = np.zeros((N,N))
for i in range(N):
    for j in range(i): # 0<=j<i
        for k in range(M):
            W[i,j] += omega(xMemory[i,k]-xMemory[j,k])
            W[j,i] += omega(xMemory[j,k]-xMemory[i,k])
sigma2_W = np.var(W.flatten())

#%%
## Define ODE
def mainode(t,x,N,W,sigma2_W,k_prior,T_theta,xTarget,x_noise):
    # Additional parameters
    # N: #neurons
    # W: Synpatic weight W[i,j] is from j to i
    # sigma2_W: variance of W
    # Generate cue
    x_tilde = xTarget + x_noise.get(t)
    k_cue = x_noise.k_cue(t)
    # Calculate phase response H
    H = np.zeros(N)
    for i in range(N):
        dxi = x[i] - x # dxi[j] = x[i] - x[j]
        H[i] = np.dot( W[i,:], domega(dxi) ) # H[i] = \sum_j W_{ij} * domega(xi-xj)
    #
    dx_prior    = -k_prior * sin(x)
    dx_external = -k_cue * sin(x-x_tilde)
    dx_synapse  = H/sigma2_W
    dx = dx_prior + dx_external + dx_synapse
    return dx
#%%
## Solve ODE
# Initial Condintion
k = 0 # memory to recall
xTarget = xMemory[:,k]
x0 = xTarget.copy() # np.random.vonmises(0,k_prior,N)
x_noise = storedNoise() # init a new random noise time trace
# Define firing events
# events = [lambda t,x,j=j: sin((x[j] - 2*pi*t/T_theta)/2) for j in range(N)]
# events[i] = 0 if and only if x[i] == 2*pi*t/T mod 2pi

# Integration
tf = T_theta*0.1
kwargs = {
    'N': N,
    'W': W,
    'k_prior': k_prior,
    'sigma2_W': sigma2_W,
    'T_theta': T_theta,
    'xTarget': xTarget,
    'x_noise': x_noise
}
# sol = solve_ivp(lambda t,y: mainode(t,y,**kwargs),(0,tf),x0,events=events)
sol = solve_ivp(lambda t,y: mainode(t,y,**kwargs),(0,tf),x0,method='BDF')
t   = sol.t; tNow = sol.t[-1]
x_t = sol.y; xNow = sol.y[:,-1]
# t_fire = sol.t_events
# x_fire = [np.mod(ts/T_theta,1)*2*pi for ts in t_fire]
print(sol.message)

#%%
# show time course
from matplotlib.cm import get_cmap
hsv = get_cmap('hsv')
ax = plt.subplot(111,projection='polar')
for xi_t,target in zip(x_t,xTarget):
    color = hsv((target/pi/2)%1)
    ax.plot(xi_t,t,color=color,alpha=0.2)
ax.plot(2*pi*t/T_theta,t,color='white')
ax.set_ylabel('time')

#%%
## evaluate errors
errors = x_t - np.transpose(np.tile(xTarget,(len(t),1)))
errors = np.mod(errors+pi,2*pi)-pi
h=plt.hist(errors[:,-1])
plt.xlim((-pi,pi))
plt.xlabel('error')
plt.ylabel('counts')

#%%
# show time course of error
ax = plt.subplot(projection='polar')
for dxi_t,target in zip(errors,xTarget):
    color = hsv((target/pi/2)%1)
    ax.plot(dxi_t,t,color=color,alpha=0.1)
ax.set_ylabel('time')
ax.set_rlabel_position(135)

#%%
# Continue Integration
tf += 10*T_theta
# sol = solve_ivp(lambda t,y: mainode(t,y,**kwargs),(tNow,tf),xNow,events=events,t_eval=np.arange(tNow,tf,5))
sol = solve_ivp(lambda t,y: mainode(t,y,**kwargs),(tNow,tf),xNow,t_eval=np.arange(tNow,tf,5),method='Radau')
t   = sol.t; tNow = sol.t[-1]
x_t = sol.y; xNow = sol.y[:,-1]
# t_fire = sol.t_events
# x_fire = [np.mod(ts/T_theta,1)*2*pi for ts in t_fire]
print(sol.message)

## evaluate errors
errors = xNow - xTarget
errors = np.mod(errors+pi,2*pi)-pi
h=plt.hist(errors)
plt.xlim((-pi,pi))
plt.xlabel('error')
plt.ylabel('counts')

#%%
