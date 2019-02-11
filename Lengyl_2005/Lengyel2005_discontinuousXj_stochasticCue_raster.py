#%%
import numpy as np
from numpy import pi,exp,sin,cos,sqrt
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
plt.style.use('default')
from scipy_ivp import solve_with_event
from scipy.interpolate import interp1d

#%%
## Define STDP and Phase coupling function
A_STDP = 0.03
s_STDP = 4
T_theta = 125 # theta oscillation period in ms
dp = lambda dt: dt*2*pi/T_theta # dt = xi - xj
omega = lambda dx: A_STDP * exp(s_STDP*cos(dx)) * sin(dx)
# derivative in respect to xi
domega = lambda dx: 2*pi/T_theta*A_STDP*exp(s_STDP*cos(dx)) * (cos(dx) - s_STDP * sin(dx)**2 )
# other parameters
N = 200 # number of neurons
M = 10 # number of memorys
k_prior = 0.5 # concentration parameter for prior distribution
k_cue = 10 # for cue distribution
k_prior = 0.5 # concentration parameter for prior distribution
k_cue0 = 16 # for initial cue distribution
v_noise = 1/80 # for cue noise accumulation, k_cue(t) = 1/( 1/k_cue0 + v_noise*t/T_theta )
tau = T_theta*8 # time constant for recall dynamics

#%% Create noise
## Create noise
class storedNoise:
    def __init__(self,dt,tf,k_cue0,v_noise):
        t = np.arange(0,tf,dt)
        nt = len(t)
        # first generate discrete noise
        xNoise_d = np.empty((nt,N))
        xNoise_d[0] = np.random.vonmises(0,k_cue0,N)
        for tt in range(nt-1):
            v = v_noise*dt/T_theta
            cumulative = np.random.normal(0,sqrt(v),N)
            xNoise_d[tt+1] = xNoise_d[tt] + cumulative
        self._t = t
        self._xNoise_d = xNoise_d
        # then interpolate with cubic spline
        self._xNoise = [interp1d(t,xNoise_d[:,ii],'cubic') 
                                for ii in range(N)]
    def __call__(self,t):
        '''
        storedNoise(t): xNoise at time t
        '''
        xNoise = np.empty(N)
        for ii in range(N):
            xNoise[ii] = self._xNoise[ii](t)
        return xNoise

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
def mainode(t,x,N,W,sigma2_W,xTarget,xNoise,k_prior,k_cue0,v_noise,T_theta,t_lastFire):
    '''    
    # Additional parameters
    # N: #neurons
    # W: Synpatic weight W[i,j] is from j to i
    # sigma2_W: variance of W
    # x_tilde is the recall cue
    # t_Lastfire is the last time when x_fires    
    '''    
    # Recalculate K_cue & x_tilde
    k_cue = 1/( 1/k_cue0 + v_noise*t/T_theta )
    x_tilde = xTarget + xNoise(t)
    # Calculate last firing phase
    t_lastFire = np.array(t_lastFire)
    x_fire = 2*pi*t_lastFire/T_theta
    fired = np.isfinite(x_fire)
    # Calculate phase response H
    H = np.zeros(N)
    for i in range(N):
        dxi = x[i] - x_fire[fired] # dxi[j] = x[i] - x[j] for j in where(fired)
        H[i] = np.dot( W[i,fired], domega(dxi) ) # H[i] = \sum_j W_{ij} * domega(xi-xj)
    # Calculate derivative
    # tau = T_theta*8 # 1000 ms
    dx_prior    = -k_prior * sin(x)
    dx_external = -k_cue * sin(x-x_tilde)
    dx_synapse  = H/sigma2_W
    dx = dx_prior + dx_external + dx_synapse
    return dx/tau
#%%
## Solve ODE
# Initial Condintion
k = 0 # memory to recall
xTarget = xMemory[:,k]
xNoise = storedNoise(1,160*T_theta,k_cue0,v_noise)
x0 = xNoise(0)+xTarget #np.random.vonmises(0,k_prior,N)

# Define firing events
events = [lambda t,x,j=j: sin((x[j] - 2*pi*t/T_theta)/2) for j in range(N)]
# events[i] = 0 if and only if x[i] == 2*pi*t/T mod 2pi

# Integration
tf = T_theta*10
t_eval = np.arange(0,tf,1)
kwargs = {
    'N': N,
    'W': W,
    'k_prior': k_prior,
    'k_cue0': k_cue0,
    'v_noise': v_noise,
    'sigma2_W': sigma2_W,
    'xNoise': xNoise,
    'xTarget': xTarget,
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
# ax.plot(2*pi*t/T_theta,t,color='gray',alpha=0.9)
for xi_t,target in zip(x_t,xTarget):
    color = hsv((target/pi/2)%1)
    ax.plot(xi_t,t,color=color,alpha=0.2)
ax.set_rgrids(range(0,tf+1,T_theta*10))
#%%
## evaluate errors
errors = x_t - np.transpose(np.tile(xTarget,(len(t),1)))
errors = np.mod(errors+pi,2*pi)-pi
#%%
plt.plot(t[t<500],errors[0,t<500])
#%%
varL = np.var(errors,axis=0)
plt.plot(t_eval[t_eval<100],varL[t_eval<100])
#%%
coolwarm = get_cmap('coolwarm')
print(coolwarm(0))
print(coolwarm(0.5))
print(coolwarm(.99))
#%% error trace
twilight_shifted = get_cmap('twilight_shifted')
def background(t,errors):
        fig,ax = plt.subplots()
        im = ax.imshow(errors,cmap='twilight_shifted',aspect='auto',vmin=-pi,vmax=pi)
        im.set_extent((t[0],t[-1],0,len(errors)))
        fig.colorbar(im)
background(t[t<2000],errors[range(50)][:,t<2000])

#%% raster
# define raster
def raster(x_fire,inds,t,errors):
        '''
        Create a raster plot and additional information in the background
        x_fire: list. Every entry is an array of firing timing (must be sorted)
        inds: iterable. index/order of neurons to plot
        t: tuple or array-like. time range to plot
        errors: background information. row indices must agree with x_fire. column must span the same range as t
        '''
        fig,ax = plt.subplots()
        background = ax.imshow(errors[inds,:],cmap='twilight_shifted',aspect='auto',vmin=-pi,vmax=pi)
        background.set_extent((t[0],t[-1],0,len(inds)))
        for iy,ii in enumerate(inds):
                for te in x_fire[ii]:
                        if te>t[-1]:
                                break
                        if te>t[0]:
                                ax.vlines(te,iy,iy+1,color='black')
        ax.set_xlim((t[0],t[-1]))
        ax.set_ylim((0,len(inds)))
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('neuron #')
        fig.colorbar(background)
ind = np.linspace(0,N-1,50).astype(int)
sortedind = np.argsort(np.mod(xTarget,2*pi))[::-1][ind]
it1 = np.where(t<T_theta*3)
raster(t_fire,sortedind,t[it1],errors[:,it1[0]])
it2 = np.where(t>T_theta*77)
raster(t_fire,sortedind,t[it2],errors[:,it2[0]])
#%% error histogram
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
sol = solve_ivp(lambda t,y: mainode(t,y,**kwargs),(tNow,tf),xNow,events=events,t_eval=np.arange(tNow,tf,5))
t   = sol.t; tNow = sol.t[-1]
x_t = sol.y; xNow = sol.y[:,-1]
t_fire = sol.t_events
x_fire = [np.mod(ts/T_theta,1)*2*pi for ts in t_fire]
print(sol.message)

## evaluate errors
errors = xNow - xTarget
errors = np.mod(errors+pi,2*pi)-pi
h=plt.hist(errors)
plt.xlim((-pi,pi))
plt.xlabel('error')
plt.ylabel('counts')

#%%
