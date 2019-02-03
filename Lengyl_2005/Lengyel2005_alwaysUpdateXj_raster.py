#%%
import numpy as np
from numpy import pi,exp,sin,cos,sqrt
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
plt.style.use('default')
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
def mainode(t,x,N,W,sigma2_W,x_tilde,k_prior,k_cue,T_theta):
    # Additional parameters
    # N: #neurons
    # W: Synpatic weight W[i,j] is from j to i
    # sigma2_W: variance of W
    # x_tilde is the recall cue
    # Calculate phase response H
    H = np.zeros(N)
    for i in range(N):
        dxi = x[i] - x # dxi[j] = x[i] - x[j]
        H[i] = np.dot( W[i,:], domega(dxi) ) # H[i] = \sum_j W_{ij} * domega(xi-xj)
    #
    tau = T_theta*4 # = 500
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
xNoise = np.random.vonmises(0,k_cue,N)
x_tilde = xTarget + xNoise
x0 = x_tilde.copy() # np.random.vonmises(0,k_prior,N)

# Define firing events
events = [lambda t,x,j=j: sin((x[j] - 2*pi*t/T_theta)/2) for j in range(N)]
# events[i] = 0 if and only if x[i] == 2*pi*t/T mod 2pi

# Integration
tf = T_theta*40
t_eval = np.arange(0,tf,0.01)
kwargs = {
    'N': N,
    'W': W,
    'k_prior': k_prior,
    'k_cue': k_cue,
    'sigma2_W': sigma2_W,
    'x_tilde': x_tilde,
    'T_theta': T_theta
}
sol = solve_ivp(lambda t,y: mainode(t,y,**kwargs),(0,tf),x0,events=events,t_eval=t_eval)
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
ax.plot(2*pi*t/T_theta,t,color='gray',alpha=0.9)
for xi_t,target in zip(x_t,xTarget):
    color = hsv((target/pi/2)%1)
    ax.plot(xi_t,t,color=color,alpha=0.2)
ax.set_rgrids(range(0,tf+1,T_theta))
#%%
## evaluate errors
errors = x_t - np.transpose(np.tile(xTarget,(len(t),1)))
errors = np.mod(errors+pi,2*pi)-pi
#%%
plt.plot(t[t<100],errors[0,t<100])
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
background(t,errors[range(50)])

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
