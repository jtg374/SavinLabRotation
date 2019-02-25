#%%
import numpy as np
import matplotlib.pyplot as plt
import brian2 as b2
#%%
from brian2 import ms,Hz,NeuronGroup,Synapses
from numpy import pi,sin,cos,exp
#%%
b2.start_scope()
# Parameters
N_s = 2**5 # Sensory Subgroup Size
nN = 8 # Number of Subgroups
N_r = 2**6 # Echo group size
tau = 10*ms # Syanaptic time constant (homogeneous)
# rate = lambda g: 0.4*(1+np.tanh(0.4*g-3))/tau # Neuron Transfer function
p = 0.35 # Exc projection probability between two groups
wFF = 2100/4 # Sensory to Echo Weight constant
wFB = 200/4  # Echo back to Sensory Weight COnstant 
A = 2; k1 = 1; k2 = 0.25 # within subgroup connection parameter
S_ext = 10 * Hz # external Input Strength
sig_ext = N_s/32 # external Input width
stim = b2.TimedArray(np.hstack((np.zeros(2),np.ones(2),np.zeros(18))),dt=50*ms) # stimlus only between 100-200 ms
# #%%
# # Transfer Func
# phi = lambda g : 0.4*(1+np.tanh(0.4*g-3))/tau
# g = np.linspace(0,12)
# plt.plot(g,phi(g)/Hz)
# #%%
eqa_N = '''
dg/dt = -g/tau + s_ext: 1 # summed synaptic variable
r = 0.4*(1+tanh(0.4*g-3))/tau: Hz # firing rate
# external input, center should be translational invariant, choose pi (N_s/2)
# note here only one subgroup included, yet to be modified
s_ext = stim(t) * S_ext * exp(-(i-N_s/2)**2/2/sig_ext**2) / sqrt(2*pi) / sig_ext : Hz
'''
PSC ='g_post += w' # w: synaptic weight variable

#%%
# Create NeuronGroups and Connections
# two Poission Neuron Groups
G   = NeuronGroup(N_s*nN,eqa_N,threshold='rand()<r*dt',name='sensory')
G_r = NeuronGroup(N_r, eqa_N,threshold='rand()<r*dt',name='random')
#%% Connection
# within subgroups
for iG in range(nN):
    start = iG*N_s
    end = start+N_s
    G_sub = G[start:end]
    S = Synapses(G_sub,G_sub,'w : 1',on_pre=PSC)
    S.connect(condition='i!=j')
    for ii in range(N_s):
        for jj in range(ii):
            u = (ii-jj)*2*pi/N_s
            w_temp = A*exp(k1*(cos(u)-1)) - A*exp(k2*(cos(u)-1))
            S.w[ii,jj] = w_temp
            S.w[jj,ii] = w_temp
# between two groups
S_FF = Synapses(G,G_r,'w : 1',on_pre=PSC) # Feedforward from sensory to echo
S_FF.connect() # connect all pairs
S_FB = Synapses(G_r,G,'w : 1',on_pre=PSC) # Feedback from echo to sensory
S_FB.connect() # connect all pairs
#%%
# random excitatory projection between two groups
C = np.random.uniform(size=(N_s*nN,N_r))<p
N_exc_s = np.sum(C,0)
N_exc_r = np.sum(C,1)
# S_FF.w[:,:] = -wFF/(N_s*nN) + wFF*C/N_exc_s[None,:] 
# S_FB.w[:,:] = -wFB/N_r    + wFB*C.T/N_exc_r[None,:]
for ii in range(N_r):
    S_FF.w[:,ii] = -wFF/(N_s*nN) + wFF*C[:,ii]/N_exc_s[ii]
for jj in range(N_s*nN):
    S_FB.w[:,jj] = -wFB/N_r    + wFB*C[jj,:]/N_exc_r[jj]
#%% verify
# print(np.sum(S_FF.w[:,np.random.choice(N_r )],0))
# print(np.sum(S_FB.w[:,np.random.choice(N_s*nN)],0))

#%%
# Monitor
sp_G = b2.SpikeMonitor(G)
r_G = b2.StateMonitor(G,'r',record=True)
g_G = b2.StateMonitor(G,'g',record=[N_s/2,N_s])
I_ext = b2.StateMonitor(G,'s_ext',record=[N_s/2,N_s])
#%%
b2.run(1100*ms)

#%%
ind = np.logical_and(sp_G.i >= N_s, sp_G.i < 2*N_s)
plt.plot(sp_G.t[ind] / ms, sp_G.i[ind], '.', markersize=2)

#%%
plt.plot(r_G.t/ms,r_G.r[0,:])
#%%
plt.plot(g_G.t/ms,g_G.g[0,:])

#%%
plt.plot(I_ext.t/ms,I_ext.s_ext[0,:])

#%%
plt.plot(r_G.r[0:N_s,np.where(r_G.t==200*ms)[0]])

#%%
