#%%
import numpy as np
from numpy import pi,exp,sin,cos
import matplotlib.pyplot as plt
from scipy_ivp import solve_with_event

#%%

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
N = 200 # number of neurons
M = 10 # number of memorys
k_prior = 0.5 # concentration parameter for prior distribution
k_cue = 10 # for cue distribution
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
def mainode(t,x,N,W,sigma2_W,x_tilde,k_prior,k_cue,t_lastFire,T_theta):
    # Additional parameters
    # N: #neurons
    # W: Synpatic weight W[i,j] is from j to i
    # sigma2_W: variance of W
    # x_tilde is the recall cue
    # t_Lastfire is the last time when x_fires
    # Calculate last firing phase
    t_lastFire = np.array(t_lastFire)
    x_fire = 2*pi*t_lastFire/T_theta
    fired = np.isfinite(x_fire)
    # Calculate phase response H
    H = np.zeros(N)
    for i in range(N):
        dxi = x[i] - x_fire[fired] # dxi[j] = x[i] - x[j] for j in where(fired)
        H[i] = np.dot( W[i,fired], domega(dxi) ) # H[i] = \sum_j W_{ij} * domega(xi-xj)
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
x0 = np.random.vonmises(0,k_prior,N)
xNoise = np.random.vonmises(0,k_cue,N)
x_tilde = xTarget + xNoise

# Define firing events
events = [lambda t,x,j=j: sin((x[j] - 2*pi*t/T_theta)/2) for j in range(N)]
# events[i] = 0 if and only if x[i] == 2*pi*t/T mod 2pi

# Integration
tf = T_theta*10
kwargs = {
    'N': N,
    'W': W,
    'k_prior': k_prior, #'Full' + 'Random initilization'
    'k_cue': k_cue,
    'sigma2_W': sigma2_W,
    'x_tilde': x_tilde,
    'T_theta': T_theta
}
odeWaitingEvent = lambda t,y,t_events_last: mainode(t,y,t_lastFire=t_events_last,**kwargs)
sol = solve_with_event(odeWaitingEvent,(0,tf),x0,events=events)
t   = sol.t; tNow = sol.t[-1]
x_t = sol.y; xNow = sol.y[:,-1]
t_fire = sol.t_events
x_fire = [np.mod(ts/T_theta,1)*2*pi for ts in t_fire]
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
# tf += 10*T_theta
# sol = solve_ivp(lambda t,y: mainode(t,y,**kwargs),(tNow,tf),xNow,events=events,t_eval=np.arange(tNow,tf,5))
# t   = sol.t; tNow = sol.t[-1]
# x_t = sol.y; xNow = sol.y[:,-1]
# t_fire = sol.t_events
# x_fire = [np.mod(ts/T_theta,1)*2*pi for ts in t_fire]
# print(sol.message)

# ## evaluate errors
# errors = xNow - xTarget
# errors = np.mod(errors+pi,2*pi)-pi
# h=plt.hist(errors)
# plt.xlim((-pi,pi))
# plt.xlabel('error')
# plt.ylabel('counts')

