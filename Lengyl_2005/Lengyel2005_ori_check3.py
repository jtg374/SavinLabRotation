#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult

#%% parameters
class param:
    A_STDP = 0.03; s_STDP = 4 # parameters for STDP
    N = 200	# number of neurons
    M = 10	# number of memories
    k_prior = 0.5	# von Mises concentration parameter
                    # for prior distribution
    k_noise = 10	# for cue distribution
    # k_synapse is determined by STDP, M and prior distribution
    tf = 320*np.pi # end of simulation, unit in LFP phase        
#%%
def Lengyel2005FullTest(nIter=10,randSeed=(None,None,None)):
    #%% Define STDP Rule and Phase Coupling Function
    # T_theta = 125 # ms, theta osicillation period
    A_STDP = param.A_STDP; s_STDP = param.s_STDP # parameters for STDP
    omega = lambda dx: A_STDP * np.exp(s_STDP*np.cos(dx)) * np.sin(dx) # gabor as STDP rule
    domega = lambda dx: A_STDP * np.exp(s_STDP*np.cos(dx)) * (np.cos(dx) - s_STDP*np.sin(dx)**2) # derivative of STDP in respect to xi (postsynaptic)
    def odesolve(t_span,x_0):
        t0,tf = t_span # end of simulation, unit in LFP phase
        #%% Define dynamics (ODEs)
        def mainode(t,x):
            # global variables:
            # N is number of neurons
            # Wij is the synpatic efficancy from neuron j to i
            # sigma2_w is the variance of W
            # x_tilde is the recall cue
            # k_{prior,noise} is the concentration of prior and 
            # cue distribution
            # x_fire is list of each neuron's list of firing 
            # phase, dynamically updated at each event
            H = np.zeros((N,N))
            for j in range(N):
                if x_fire[j]: # if neuron j has fired
                    xj = x_fire[j][-1] # we need the last firing phase
                    dxij = x-xj # phase difference between neurons
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
        event = [lambda t,x,j=j: np.sin((x[j]-t)/2) for j in range(N)]      
                # event[i](t,x)
                # equals 0 when firing phase
                # of neuron i matches current LFP 
                # i.e. (x[i]-t) mod 2pi ==0   
                # syntax: j=j freeze j google "python define list of function"
        #%%
        x_fire = [[ ] for j in range(N)] # record firing phase
        for func in event: 
                func.terminal = True # stop integration when any neuron fires
        #%% solve ode while recording firing
        #%% first round
        sol = solve_ivp(mainode,(t0,tf),x_0,events=event) # integrate
                                        # until a neuron fire
        t = sol.t; tNow = t[-1]         # record time
        x = sol.y; xNow = x[:,-1]       # state
        for j,te in enumerate(sol.t_events):
                if te.size>0:
                        # print(j)
                        x_fire[j] += list(te)      # update x_fire for calculating H
        # print(sol.message)
        # print(tNow)
        #%% second and subsequenct round
        while tNow < tf:
                sol = solve_ivp(mainode,(tNow,tf),xNow,events=event)
                # t = np.append(t,sol.t)          # integrate until one neuron
                # x = np.append(x,sol.y,axis=1)   # fires, record time, state,
                tNow = sol.t[-1]                    # firing time and neuron
                xNow = sol.y[:,-1]                  # index
                whoFire = []
                for j,te in enumerate(sol.t_events): 
                        event[j].terminal = True 
                        if te.size>0:
                                event[j].terminal = False
                                if (not x_fire[j]) or te[0]>x_fire[j][-1]+1e-5: # 'refractory' to prevent overcounting spikes
                                        whoFire += [j]
                                        x_fire[j] += list(te)  # update x_fire for calculating H
                if 0 in whoFire: # if neuron 0 fires
                        print(sol.message)
                        print(tNow) # report current time
        return xNow
    #%% main
    N = param.N	# number of neurons
    M = param.M	# number of memories
    memorys = np.empty((nIter,N,M))
    cues = np.empty((nIter,N,M))
    synapses = np.empty((nIter,N,N))
    results = np.empty((nIter,N,M))
    for mainIter in range(nIter):
        print('iter: ',mainIter)        
        #%% initialize memory traces and synaptic weights
        k_prior = param.k_prior	# von Mises concentration parameter
                                # for prior distribution
        k_noise = param.k_noise	# for cue distribution
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
        memorys[mainIter] = x_memory
        synapses[mainIter] = W
        #%% 
        for k in range(M):      # the kth memory trace is
            print('memory#',k)
            x_target = x_memory[:,k].copy()     # what we want to recall
            x_noise = np.random.vonmises(0,k_noise,N)   # indepedent random
                                            # noise to corrupt the cue
            x_tilde = x_target + x_noise 
            cues[mainIter][:,k] = x_tilde
            x_0 = np.random.vonmises(0,k_prior,N) # initial condition
            xFinal = odesolve(t_span=(0,param.tf),x_0=x_0)
            results[mainIter][:,k] = xFinal
    return memorys,cues,synapses,results
#%%
memorys,cues,synapses,results = Lengyel2005FullTest()
np.save('xMemoryList_1230',memorys)
np.save('xCueList_1230',cues)
np.save('WList_1230',synapses)
np.save('xFinalList_1230',results)
