#%%
import numpy as np
from numpy import pi,sin,cos,exp,sqrt
import matplotlib.pyplot as plt
#%% Parameters
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


#%% Load data
# load data
t = np.arange(0,tf,dt)
len_t = len(t)
xMemory = np.empty((nIter,N,M))
xRecalled = np.empty((nIter,N,len_t,M))
xRecalled_n = np.empty((nIter,N,len_t,M))
xNoise = np.empty((nIter,N,len_t+1,M))
for iIter in range(nIter):
    filename = 'Data/Lengyel2005_alwaysUpdateXj_stochasticCue/Lengyel2005_alwaysUpdateXj_stochasticCue_withDynamicWeight_iter%02d.npz'%(iIter)
    loaded = np.load(filename)
    # print(loaded.files)
    xMemory[iIter] = loaded['xMemory']
    xRecalled[iIter] = loaded['xRecalled']
    xNoise[iIter] = loaded['xNoise']
for iIter in range(nIter):
    filename = 'Data/Lengyel2005_alwaysUpdateXj_stochasticCue/Lengyel2005_alwaysUpdateXj_stochasticCue_noDynamicWeight_iter%02d.npz'%(iIter)
    loaded = np.load(filename)
    xRecalled_n[iIter] = loaded['xRecalled']



#%% with dynamic weight
# Error of result simulated with dynamic weight
xError = np.empty_like(xRecalled)
for tt in range(len_t):
    xError[:,:,tt,:] = xRecalled[:,:,tt,:] - xMemory
xError = np.swapaxes(xError,2,3)
xError = np.reshape(xError,(nIter*N*M,len_t),order='C')
xError = np.mod(xError+pi,pi*2) - pi

#%% without dynamic weight
# Error of result simulated without dynamic weight
xError_n = np.empty_like(xRecalled_n)
for tt in range(len_t):
    xError_n[:,:,tt,:] = xRecalled_n[:,:,tt,:] - xMemory
xError_n = np.swapaxes(xError_n,2,3)
xError_n = np.reshape(xError_n,(nIter*N*M,len_t),order='C')
xError_n = np.mod(xError_n+pi,pi*2) - pi


#%% Control level
# Control level (FF only)
k_cue = 1/ (1/k_cue0 + v_noise*t/T_theta)
xError_FF = np.empty_like(xError)
for tt in range(len_t):
    xCue = xNoise[:,:,tt,:] + xMemory
    Z = exp(1j*xCue) + k_prior/k_cue[tt]
    xFF = np.angle(Z)
    Error_t = xFF-xMemory
    xError_FF[:,tt] = Error_t.flatten()
xError_FF = np.mod(xError_FF+pi,pi*2) - pi

#%% save
filename = 'Data/Lengyel2005_alwaysUpdateXj_stochasticCue/ErrorCombined.npz'
np.savez(filename,xError=xError,xError_FF=xError_FF,xError_n=xError_n)

#%% load
filename = 'Data/Lengyel2005_alwaysUpdateXj_stochasticCue/ErrorCombined.npz'
loaded = np.load(filename)
xError = loaded['xError']
xError_n = loaded['xError_n']
xError_FF = loaded['xError_FF']

#%% stats time course 
xError_all = np.array([xError,xError_n,xError_FF])
labels = ['dynamic cue weight','fixed cue weight','no synaptic info']
bias = np.mean(xError_all,axis=1)
stdL = np.std(xError_all,axis=1)
dspL = stdL**2 + bias**2
vR   = exp(1j*xError_all)
R    = abs(np.mean(vR,axis=1))
varC = 1-R
dspC = varC + 2*R*(np.sin(bias/2)**2)

#%% save summary stats
filename = 'Data/Lengyel2005_alwaysUpdateXj_stochasticCue/Stats.npz'
np.savez(filename,bias=bias,stdL=stdL,dspL=dspL,varC=varC,dspC=dspC)
#%%
print(plt.style.available)
plt.style.use('seaborn')
plt.style.use('seaborn-talk')

#%%
tt = int(1/dt)
hist = [None for _ in range(3)]
fig1,ax1 = plt.subplots()
for m in range(3):
    hist[m],bin_edges,_ = ax1.hist(xError_all[m][:,tt],200,(-pi,pi), label=labels[m])
fig1.legend()
fig2 = plt.figure()
ax2 = fig2.add_subplot(111,projection='polar')
for m in range(3):
    ax2.plot(bin_edges,np.append(hist[m],hist[m][0]),drawstyle='steps-pre',label=labels[m])
fig2.legend()

#%%
print(dspL[0][:100])

#%%
fig,ax = plt.subplots()
for m in range(3):
    ax.plot(t,dspL[m],label=labels[m])
ax.set_xlabel('time (ms)')
ax.set_ylabel('linear mean squared error')
fig.legend()
#%%
