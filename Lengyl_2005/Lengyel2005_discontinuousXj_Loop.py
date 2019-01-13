#%%
import numpy as np
from numpy import pi,exp,sin,cos
import matplotlib.pyplot as plt
from scipy_ivp import solve_with_event
from datetime import datetime
import pickle

#%%
# Global Experimental parameters
nIter = 10
N = 200 # number of neurons
M = 10 # number of memorys, every trace will be attemped to recall
T_theta = 125 # theta oscillation period in ms
tf = 5*T_theta # integration time for each recall
dt = 1 # timestep for saving results

#%%
# Main loop
for iIter in range(nIter):
    print('Iternation #',iIter+1)

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

    ## Prepare space for saving results
    t_eval = np.arange(0,tf,dt)
    len_t = len(t_eval)
    cues = np.empty((N,M))
    recalled = np.empty((N,len_t,M))
    t_fireList = []
    #%%
    ## Solve ODE
    for k in range(M): # every memory trace will be attempted to recall
        print('memory #',k+1,'/',M)
        # Initial Condintion
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
        # save result for current recall
        cues[:,k]= x_tilde
        recalled[:,:,k] = x_t
        t_fireList += [t_fire]
    #%%
    # save all into file
    now = datetime.now()
    filename = 'Lengyel2005_discontinuousXj_iter%02d'%(iIter)
    np.savez(filename,xMemory=xMemory,W=W,
        xCues=cues,xRecalled=recalled,
        time=now,t_eval=t_eval)
    with open(filename+'_firing.data','wb') as f:
        pickle.dump(t_fireList,f)
