#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.integrate import solve_ivp

#%% Define STDP Rule and Phase Coupling Function
# T_theta = 125 # ms, theta osicillation period
A_STDP = 0.03; s_STDP = 4 # parameters for STDP
omega = lambda dx: A_STDP * np.exp(s_STDP*np.cos(dx)) * np.sin(dx) # gabor as STDP rule
domega = lambda dx: A_STDP * np.exp(s_STDP*np.cos(dx)) * (np.cos(dx) - s_STDP*np.sin(dx)**2) # derivative of STDP in respect to xi (postsynaptic)
# plot STDP and PCF
# u = np.arange(-np.pi,np.pi,np.pi/120)
# plt.plot(u,omega(u))
# plt.plot(u,domega(u))
# plt.plot(u,domega(u)*omega(u))
#%% Plot Phase Response Curve
du=np.pi/120 
u = np.arange(-np.pi,np.pi,du) # a bunch of testing phase 
uu=u.copy();uu[u<=0]+=2*np.pi # wrap into range (0,2pi]
PRC = np.zeros_like(u)
k_prior = 0.6 # confidence of prior (x=0)
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
#%% Define dynamics
def mainode(t,x):
    # global variables:
    # N is number of neurons
    # Wij is the synpatic efficancy from neuron j to i
    # sigma2_w is the variance of W
    # x_tilde is the recall cue
    # k_{prior,noise} is the concentration of prior and 
    # cue distribution
    # phase, dynamically updated at each event
    H = np.zeros((N,N))
    for j in range(N):
        dxij = x-x[j] # phase difference between neurons
        H[:,j] = W[:,j] * domega(dxij) # phase interaction
                                        # from neuron j
                                        # to neuron i
                                        # Hjj is guarenteed
                                        # to be 0 because
                                        # Wjj is 0
    dx_prior    = -k_prior * np.sin(x) 
    dx_external = -k_noise * np.sin(x-x_tilde)
    dx_synapse  = np.sum(H,1)/sigma2_w # sum_j H_{ij}/sigma_w^2
    return dx_prior + dx_external + dx_synapse

#%% Define firing events
N = 200	# number of neurons
event = [lambda t,x,j=j: np.sin((x[j]-t)/2) for j in range(N)]      
        # event[i](t,x)
        # equals 0 when firing phase
        # of neuron i matches current LFP 
        # i.e. (x[i]-t) mod 2pi ==0   
        # syntax: j=j freeze j google "python define list of function"                              
#%% Parameters and initial conditions
N = N	# number of neurons
M = 10	# number of memories
k_prior = 0.5	# von Mises concentration parameter
                # for prior distribution
k_noise = 10	# for cue distribution
x_memory = np.random.vonmises(0,k_prior,(N,M)) # every column 
                                        # is a memory trace
W = np.zeros((N,N))
for i in range(N):
    for j in range(i): # j<i, Wii = 0
        for k in range(M): # the STDP rule is additive
            W[i,j] += omega(x_memory[i,k]-x_memory[j,k])
            W[j,i] += omega(x_memory[j,k]-x_memory[i,k])
W_flatten = [W[i][j] for i in range(N) for j in range(i) ]
sigma2_w = np.var(W_flatten)
# plt.hist(W_flatten,50)
#%% initial conditions
kk=0
x_target = x_memory[:,kk].copy()     # the kk'th one is 
                                    # what we want to recall
x_0 = np.random.vonmises(0,k_prior,N) # random initial state
x_noise = np.random.vonmises(0,k_noise,N)   # indepedent random
                                # noise to corrupt the cue
x_tilde = x_target + x_noise 

# x_fire = [[ ] for j in range(N)] # record firing phase

#%% sanity checks for W
print(not np.any(np.diag(W)))
i,j = np.random.randint(0,N,2)
print(W[i,j] + W[j,i] == 0)
#%% Which one fires first? 
# p=np.argmin(np.mod(x_0,2*np.pi))
# print(p)
# print(x_0[p])
# print(x_target[p])
# print(x_tilde[p])
#%% Solve ODE while detecting events
for func in event: 
        func.terminal = False # continue integration when neuron fires
# first round
tf = 20*np.pi # end of simulation, unit in LFP phase
sol = solve_ivp(mainode,(0,tf),x_0,events=event) # integrate
                                # until the end
t = sol.t; tNow = t[-1]         # record time
x = sol.y; xNow = x[:,-1]       # state
x_fire = sol.t_events
print(sol.message)
print(tNow)
# print(len(xNow))
#%% continue
tf += 20*np.pi # end of simulation, unit in LFP phase
sol = solve_ivp(mainode,(tNow,tf),xNow,events=event) # integrate
                                # until the end
t = np.append(t,sol.t); tNow = t[-1]         # record time
x = np.append(x,sol.y); xNow = x[:,-1]       # state
x_fire = sol.t_events
print(sol.message)
print(tNow)
#%%
# sol = solve_ivp(mainode,(tNow,tf),xNow,events=event)
# print(len(sol.t_events))
# t = np.append(t,sol.t)
# x = np.append(x,sol.y,axis=1)
# tNow = t[-1]
# xNow = x[:,-1]
# print(sol.message)
# print(tNow)
# for j,te in enumerate(sol.t_events):
#         if te.size>0:
#                 print(j)
#%% Time course of a single neuron
p=3
plt.plot(t,x[p,:])
# #%% Fire at right phase? 
# p = p
# te = list(sol.t_events[p])
# print(te)
# tFire = te[-1]
# print(event[p](tFire,x[:,t==tFire][:,0]))
# print(x[p,t==tFire][0])
#%% second and subsequenct round
# while tNow < tf:
#         sol = solve_ivp(mainode,(tNow,tf),xNow,events=event)
#         t = np.append(t,sol.t)          # integrate until one neuron
#         x = np.append(x,sol.y,axis=1)   # fires, record time, state,
#         tNow = t[-1]                    # firing time and neuron
#         xNow = x[:,-1]                  # index
#         whoFire = []
#         for j,te in enumerate(sol.t_events): 
#                 event[j].terminal = True 
#                 if te.size>0:
#                         event[j].terminal = False
#                         if te[0]>x_fire[j][-1]+1e-5: # 'refractory' to prevent overcounting spikes
#                                 whoFire += [j]
#                                 x_fire[j] += list(te)  # update x_fire for calculating H
#         print(sol.message)
#         print(whoFire)
#         print(tNow)
#%%
t<=t2 & t>=t1
#%%
p=87
print(x_fire[p])
t1 = np.where(t==x_fire[p][0])[0]
t2 = np.where(t==x_fire[p][1])[0]
print(t1)
print(t2)
#%%

plt.plot(t[(t1-1):(t2+2)],x[p,(t1-1):(t2+2)])
#%% raster
def raster(x_fire,inds):
        nn = len(inds)
        fig,ax = plt.subplots()
        for i in inds:
                for te in x_fire[i]:
                        ax.vlines(te,i,i+1,color='white')
        ax.set_ylim(0,nn)
        ax.set_xlabel('time in LFP cycle')
        ax.set_ylabel('neuron')
        xticks = np.arange(0,tf+2*np.pi,2*np.pi)
        ax.set_xticks(xticks)
        ax.set_xticklabels(map(str,range(len(xticks))))
        # fig.show()
        return fig
raster(x_fire,range(10))
# inds = range(10)
# N = len(inds)
# for i in inds:
#         for te in x_fire[i]:
#                 plt.vlines(te,i,i+1,color='white')
# plt.ylim(0,N)
# plt.xlabel('time in LFP phase')
# plt.ylabel('neuron')
#%% simple plot to visualize time course
for i in range(N):
        p = x_target[i]
        plt.plot(np.mod(x_fire[i],np.pi/2),c=cm.hsv(p/(2*np.pi)))

#%%
for i in range(10):
        print(i)
        print(len(x_fire[i]))
#%% evaluate errors
finalPhase = [ x_fire[i][-1] for i in range(N) ]
errors = np.array(finalPhase) - x_target
errors = np.mod(errors+np.pi,2*np.pi)-np.pi
plt.hist(errors,20)
plt.xlabel('errors')
plt.ylabel('counts')
#%%
t=np.load('t_12112151.npy')
x=np.load('x_12112151.npy')
x_fire=np.load('x_fire_12112151.npy')
#%%
