#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
def ff(cue,kratio):
    res = np.exp(1j*cue) + kratio
    return np.angle(res)
def circR(samples,axis=None):
    res = np.mean(np.exp(1j * samples), axis=axis)
    return res
def wrap(ang):
    return np.mod(ang+np.pi,2*np.pi) - np.pi
#%%
import pickle
with open('firing_1230.data','rb') as f:
    solutions=pickle.load(f)
memorys = np.load('xMemoryList_1230.npy')
memorys = np.mod(memorys,np.pi*2)
synapses = np.load('WList_1230.npy')
cues = np.load('xCueList_1230.npy')
results = np.load('xFinalList_1230.npy')
results = np.mod(results,np.pi*2)
#%%
#%%
# del memorys
# del cues
# del synapses
# del results
# #%%
# del solutions
#%%
nIter = 3
M = 15
N= 200
modes = ['Full','Feedforward','None']
#%%
# len(solutions)
# #%%
# results = np.empty((nIter,N,M))
# for ii in range(nIter):
#     for k in range(M):
#         results[ii][:,k] = solutions[M*ii+k]['final']
#%%
# from matplotlib import cm
# ii = np.random.randint(0,nIter)
# k = np.random.randint(0,M)
# ind = M*ii+k
# x_target = memorys[ii][:,k]
# x_t = solutions[ind]['y']
# t = solutions[ind]['t']
# for i in np.random.randint(0,N,20):
#     p = np.mod(x_target[i],2*np.pi)
#     plt.plot(t,x_t[i,:],c=cm.hsv(p/(2*np.pi)))
# # plt.colorbar()
#%%
x = {}
x[modes[0]] = results
k_prior = 0.5	# von Mises concentration parameter
                # for prior distribution
k_noise = 10	# for cue distribution
x[modes[1]] = ff(cues,k_prior/k_noise)
x[modes[2]] = np.zeros_like(results)
#%%
dx = {}
for mode in modes:
    dx[mode] = x[mode]-memorys
    dx[mode] = dx[mode].flatten()
    dx[mode] = wrap(dx[mode])
#%%
h = {}
for mode in modes:
    h[mode] = plt.hist(dx[mode],100,(-np.pi,np.pi),density=True,histtype='step',label=mode)
    plt.legend()
        
#%%
R = {}
bias= {}
var = {}
lsd = {}
error = {}
# S = {}
for mode in modes:
    lsd[mode] = np.std(dx[mode])
    R[mode] = circR(dx[mode])
    bias[mode] = np.angle(R[mode])
    var[mode] = 1 - abs(R[mode])
    error[mode] = var[mode] + 2*abs(R[mode])*(np.sin(bias[mode]/2)**2)
    # h[mode] = h[mode]/np.sum(h[mode])
    # S[mode] = -np.sum(h[mode]*np.log(h[mode]))
print('mean bias: \n',bias)
print('linear root mean squared error: \n',lsd)
print('circular variance: \n',var)
print('circular dispersion: \n',error)
# print('Entropy: ',S)
#%%
from scipy.special import iv
print('var_none_theoretical:',1-iv(1,k_prior)/iv(0,k_prior))

