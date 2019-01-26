#%%
import numpy as np
from numpy import pi,exp,sin,cos,sqrt
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from datetime import datetime
import pickle
from scipy.interpolate import interp1d

#%%
# Global Experimental parameters
nIter = 10
N = 200 # number of neurons
M = 10 # number of memorys, every trace will be attemped to recall
T_theta = 125 # theta oscillation period in ms
tf = 10*T_theta # integration time for each recall
dt = 1 # timestep for saving results
k_prior = 0.5 # concentration parameter for prior distribution
k_cue0 = 16 # for initial cue distribution
v_noise = 1/8 # for cue noise accumulation, k_cue(t) = 1/( 1/k_cue0 + v_noise*t/T_theta )

#%% Create noise
## Create noise
class storedNoise:
    def __init__(self,dt,tf,xNoise_d):
        t = np.arange(0,tf,dt)
        self._t = t
        self._xNoise_d = xNoise_d
        # interpolate with cubic spline
        self._xNoise = [interp1d(t,xNoise_d[ii,:],'cubic') 
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
# Main loop
for iIter in range(nIter):
    print('Iternation #',iIter+1)
    inputFilename = 'Data/Lengyel2005_alwaysUpdateXj_stochasticCue/Lengyel2005_alwaysUpdateXj_stochasticCue_iter%02d.npz'%(iIter)
    loaded = np.load(inputFilename)
    #%%
    ## Define STDP and Phase coupling function
    T_theta = T_theta # theta oscillation period in ms
    A_STDP = 0.03
    s_STDP = 4
    dp = lambda dt: dt*2*pi/T_theta # dt = xi - xj
    omega = lambda dx: A_STDP * exp(s_STDP*cos(dx)) * sin(dx)
    # derivative in respect to xi
    domega = lambda dx: 2*pi/T_theta*A_STDP*exp(s_STDP*cos(dx)) * (cos(dx) - s_STDP * sin(dx)**2 )

    #%%
    ## Create Memorys
    N = N # number of neurons
    M = M # number of memorys
    xMemory = loaded['xMemory']
    ## Create Synapses
    W = loaded['W']
    sigma2_W = np.var(W.flatten())

    #%%
    ## Define ODE
    def mainode(t,x,N,W,sigma2_W,xTarget,xNoise,k_prior,k_cue0,v_noise,T_theta):
        # Additional parameters
        # N: #neurons
        # W: Synpatic weight W[i,j] is from j to i
        # sigma2_W: variance of W
        # x_tilde is the recall cue
        # Recalculate K_cue
        # k_cue = 1/( 1/k_cue0 + v_noise*t/T_theta )
        x_tilde = xTarget + xNoise(t)
        # Calculate phase response H
        H = np.zeros(N)
        for i in range(N):
            dxi = x[i] - x # dxi[j] = x[i] - x[j]
            H[i] = np.dot( W[i,:], domega(dxi) ) # H[i] = \sum_j W_{ij} * domega(xi-xj)
        #
        tau = T_theta*0.08 # = 10
        dx_prior    = -k_prior * sin(x)
        dx_external = -k_cue0 * sin(x-x_tilde)
        dx_synapse  = H/sigma2_W
        dx = dx_prior + dx_external + dx_synapse
        return dx/tau
    
    ## Prepare space for saving results
    t_eval = np.arange(0,tf,dt)
    len_t = len(t_eval)
    recalled = np.empty((N,len_t,M))
    t_fireList = []
    #%%
    ## Solve ODE
    for k in range(M): # every memory trace will be attempted to recall
        print('memory #',k+1,'/',M)
        # Initial Condintion
        xTarget = xMemory[:,k]
        xNoise_d = loaded['xCue'][:,:,k] # due to previous mistake... naming mess
        xNoise = storedNoise(dt,tf+dt,xNoise_d)
        x0 = xNoise(0) + xTarget

        # Define firing events
        events = [lambda t,x,j=j: sin((x[j] - 2*pi*t/T_theta)/2) for j in range(N)]
        # events[i] = 0 if and only if x[i] == 2*pi*t/T mod 2pi

        # Integration
        tf = tf
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
        sol = solve_ivp(lambda t,y: mainode(t,y,**kwargs),(0,tf),x0,events=events,t_eval=t_eval)
        t   = sol.t; tNow = sol.t[-1]
        x_t = sol.y; xNow = sol.y[:,-1]
        t_fire = sol.t_events
        # x_fire = [np.mod(ts/T_theta,1)*2*pi for ts in t_fire]
        print(sol.message)

        #%% 
        # save result for current recall
        recalled[:,:,k] = x_t
        t_fireList += [t_fire]
    #%%
    # save all into file
    now = datetime.now()
    outputFilename = 'Data/Lengyel2005_alwaysUpdateXj_stochasticCue/Lengyel2005_alwaysUpdateXj_stochasticCue_noDynamicWeight_iter%02d.npz'%(iIter)
    np.savez(outputFilename,
        xRecalled=recalled,
        time=now,t_eval=t_eval)
    with open(outputFilename+'_firing.data','wb') as f:
        pickle.dump(t_fireList,f)

