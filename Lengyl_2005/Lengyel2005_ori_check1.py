#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.integrate import solve_ivp

#%% parameters
class param:
    A_STDP = 0.03; s_STDP = 4 # parameters for STDP
    N = 200	# number of neurons
    M = 10	# number of memories
    k_prior = 0.5	# von Mises concentration parameter
                    # for prior distribution
    k_noise = 10	# for cue distribution
    # k_synapse is determined by STDP, M and prior distribution
    tf = 10*np.pi # end of simulation, unit in LFP phase        
#%%
def Lengyel2005FullTest(nIter=10,randSeed=(None,None,None)):
    #%% Define STDP Rule and Phase Coupling Function
    # T_theta = 125 # ms, theta osicillation period
    A_STDP = param.A_STDP; s_STDP = param.s_STDP # parameters for STDP
    omega = lambda dx: A_STDP * np.exp(s_STDP*np.cos(dx)) * np.sin(dx) # gabor as STDP rule
    domega = lambda dx: A_STDP * np.exp(s_STDP*np.cos(dx)) * (np.cos(dx) - s_STDP*np.sin(dx)**2) # derivative of STDP in respect to xi (postsynaptic)
    def odesolve(t_span,x_0,mode='full'):
        print(mode)
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
                dxij = x-x[j] # phase difference between neurons
                H[:,j] = W[:,j] * domega(dxij) # phase interaction
                                            # from neuron j
                                            # to neuron i
                                            # Hjj is guarenteed
                                            # to be 0 because
                                            # Wjj is 0
            dx_prior    = -k_prior * np.sin(x)
            if mode is 'full':
                dx_external = -k_noise * np.sin(x-x_tilde)
                dx_synapse  = np.sum(H,1)/sigma2_w # sum_j H_{ij}/sigma_w^2
                return dx_prior + dx_external + dx_synapse
            elif mode is 'feedforward':
                dx_external = -k_noise * np.sin(x-x_tilde)
                return dx_prior + dx_external
            elif mode is 'none':
                return dx_prior
        #%% Define firing events
        event = [lambda t,x,j=j: np.sin((x[j]-t)/2) for j in range(N)]      
                # event[i](t,x)
                # equals 0 when firing phase
                # of neuron i matches current LFP 
                # i.e. (x[i]-t) mod 2pi ==0   
                # syntax: j=j freeze j google "python define list of function"
        #%%
        for func in event: 
                func.terminal = False # stop integration when any neuron fires
        #%% solve ode while recording firing
        #%% first round
        sol = solve_ivp(mainode,(t0,tf),x_0,events=event) # integrate
        #                                 # until a neuron fire
        # t = sol.t; tNow = t[-1]         # record time
        # x = sol.y; xNow = x[:,-1]       # state
        # x_fire = sol.t_events
        # print(sol.message)
        # print(tNow)
        return sol
    #%% main
    memorys = []
    synapses = []
    solution = []
    for mainIter in range(nIter):
        results = []
        print('iter: ',mainIter)        
        #%% initialize memory traces and synaptic weights
        N = param.N	# number of neurons
        M = param.M	# number of memories
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
        memorys += [x_memory]
        synapses += [W]
        #%% 
        for k in range(M):      # the kth memory trace is
            print('memory#',k)
            x_target = x_memory[:,k].copy()     # what we want to recall
            x_noise = np.random.vonmises(0,k_noise,N)   # indepedent random
                                            # noise to corrupt the cue
            x_tilde = x_target + x_noise 
            x_0 = np.random.vonmises(0,k_prior,N) # initial condition
            solFull = odesolve(t_span=(0,param.tf),x_0=x_0,mode='full')
            solFF = odesolve(t_span=(0,param.tf),x_0=x_0,mode='feedforward')
            solPrior = odesolve(t_span=(0,param.tf),x_0=x_0,mode='none')
            sol = {'Full':solFull,'Feedforward':solFF,'None':solPrior}
            results += [sol]
        solution += [results]
    return memorys,synapses,solution
#%%
memorys,synapses,solution = Lengyel2005FullTest(10)
np.save('xMemoryList_1229',memorys)
np.save('WList_1229',synapses)
#%%
import pickle
with open('firing_1229.data','wb') as f:
    pickle.dump(solution,f)

#%%
def A(a,b):
    return a*b+c
def B(func,a,b):
    return func(a,b)
a = np.array([2,3])
b = np.array([2,4])
c = np.array([5,3])
print(B(A,3,4))
c = np.array([6,4])
print(B(A,3,4))