#%%
import numpy as np
import matplotlib.pyplot as plt

#%% Phase Coupling Function
T_theta = 125 # ms, theta osicillation period
A_STDP = 0.03; s_STDP = 4 # parameters for STDP
omega = lambda dx: A_STDP * np.exp(s_STDP*np.cos(dx)) * np.sin(dx) # gabor as STDP rule
domega = lambda dx: A_STDP * np.exp(s_STDP*np.cos(dx)) * (np.cos(dx) - s_STDP*np.sin(dx)**2) # derivative of STDP in respect to xi (postsynaptic)
# plot STDP and PCF
# u = np.arange(-np.pi,np.pi,np.pi/120)
# plt.plot(u,omega(u))
# plt.plot(u,domega(u))
# plt.plot(u,domega(u)*omega(u))
#%% Phase Response curve
du=np.pi/120 
u = np.arange(-np.pi,np.pi,du) # a bunch of testing phase 
uu=u.copy();uu[u<=0]+=2*np.pi # wrap into range (0,2pi]
PRC = np.zeros_like(u)
k_prior = 0.6
for w in [0.01,0.025, 0.05, 0.075]:
        for i,xj in enumerate(uu):
                x=0 # postsynaptic start from 0 phase
                t=xj # phase response integrate from presynaptic fire
                dt = 0.002*np.pi
                while (t+dt-x)<2*np.pi: # fire when x == t mod 2pi
                        dx = dt*(-k_prior*np.sin(x)+w*domega(x-xj))
                        x+=dx
                        t+=dt
                PRC[i] = x
        plt.plot(u,PRC)

#%% memory encoding
N = 200 # num of neurons
M = 15 # number of memories
# generate memories
k_prior = 0.5
x = np.random.vonmises(0,k_prior,(N,M)) # draw random memory from von mises distribution
# every column is a stored memory
x = x%(2*np.pi) # so that phase go from 0 to 2pi
# x[:,0] = x[:,0].sort()
# print(x)
# encode into synpases
W = np.zeros((N,N))
for i in range(N):
    for j in range(i+1,N):
        for k in range(M):
            W[i,j] += omega(x[i,k]-x[j,k])
for i in range(N):
    for j in range(i):
        W[i,j] = -W[j,i]
# print(W)        
# plt.hist(W.flatten(),100)
# W/= np.std(W) # normalize by standard deviation so that variance of w is 1
sigma_w = np.std(W) # estimate from samples
# plt.imshow(W)
# plt.colorbar()

#%% Memory recall
# cue and initial condition
x_0 = x[:,0] # start from exact attractor for stability check
# x_0 = np.random.vonmises(0,k_prior,N) # initial state before recall
k_cue = 10 # 1/rad^2
k_noise = 100 # Hz/rad^2
x_noise = np.random.vonmises(0,k_cue,N)
x_tilde = x[:,0] # start from exact attractor for stability check # + x_noise
x_tilde_0 = x_tilde.copy()
# numerical integrator
## timing
tau_x = 1 # rad
dt = 0.02*np.pi # # time is in unit of theta phase
T = 4*np.pi
## initial condition and recording output
x_t = x_0.copy()
x_fire = [[x_t[i]] for i in range(N)] # allocate for recording spiking timing (phase)
t_fire = [[] for i in range(N)]
# for i in range(N):
#         for j in range(i):
#                 H[i,j] = W[i,j]*domega(x_t[i]-x_t[j])
#                 H[j,i] = -H[i,j]
for t in np.arange(0,T,dt): # time is in unit of theta phase
        H = np.zeros((N,N)) # unit interaction
        for j in range(N):
                if x_fire[j]:
                        xj = x_fire[j][-1]
                        for i in range(N):
                                H[i,j] = W[i,j]*domega(x_t[i]-xj)
        Hi = np.sum(H,1) 
        # x_noise = np.random.normal(0,1/np.sqrt(k_noise),N)
        dx = dt/tau_x * (-k_prior*np.sin(x_t) + k_cue*np.sin(x_tilde-x_t) + Hi/sigma_w**2) #shit!! Hi should depend on xi(t)
        nFire_before = np.floor((t-x_t)/2/np.pi)
        nFire_after = np.floor((t+dt-x_t-dx)/2/np.pi)
        for j in range(N):
                if nFire_after[j]>nFire_before[j]: # fire when x == t mod 2pi
                        xj = x_t[j] + dx[j]*(nFire_after[j]*2*np.pi - (x_t[j]+t))/(dx[j]+dt)
                        tj = t      + dt   *(nFire_after[j]*2*np.pi - (x_t[j]+t))/(dx[j]+dt)
                        # record firing time
                        x_fire[j].append(xj%(2*np.pi))
                        t_fire[j].append(tj)
        x_t += dx
#%% Evaluate error
finalPhase = [x_fire[i][-1] for i in range(N)]
finalPhase = np.array(finalPhase)
error = finalPhase - x[:,0]
error = np.arccos(np.cos(error))
plt.hist(error)
#%% Visualize time course
from matplotlib import cm
for i in range(N):
        p = x[i,0]
        plt.plot(x_fire[i],c=cm.hsv(p/(2*np.pi)))
#%%
plt.hist(Hi)
#%%
## postsynaptic effect at attractor
N = 200 # num of neurons
M = 15 # number of memories
k_prior = 0.5
x = np.random.vonmises(0,k_prior,(N,M)) # draw random memory from von mises distribution # every column is a stored memory
# x[:,0] = np.linspace(0,2*np.pi,N)
W = np.zeros((N,N))
for i in range(N):
    for j in range(i+1,N):
        for k in range(M):
            W[i,j] += omega(x[i,k]-x[j,k])
for i in range(N):
    for j in range(i):
        W[i,j] = -W[j,i]
sigma_w = np.std(W) # estimate from samples

#
x_t = x[:,0]
H = np.zeros((N,N)) # unit interaction
for i in range(N):
        for j in range(i):
                H[i,j] = W[i,j]*domega(x_t[i]-x_t[j])
                H[j,i] = -H[i,j]
H /= sigma_w**2
Hi = np.sum(H,0) # sum into postsynaptic effects
plt.hist(Hi,50)
#%%
N = 200
k_prior = 0.5
sigma_w = 1
x = np.random.vonmises(0,k_prior,N-1) # fix other neurons
His = []
u = np.arange(-np.pi,np.pi,np.pi/120) # systematically change postsynaptic neuron
for x0 in u:
        Wj = (omega(x-x0) + np.random.randn(N-1)*sigma_w)
        Hj = domega(x-x0) * Wj
        His.append(np.sum(Hj))
plt.plot(u,His)
