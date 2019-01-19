#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
def circR(samples,axis=None):
    res = np.mean(np.exp(1j * samples), axis=axis)
    return res
def wrap(ang):
    return np.mod(ang+np.pi,2*np.pi) - np.pi
#%%
import pickle
with open('firing_1226.data','rb') as f:
    results=pickle.load(f)
memorys = np.load('xMemoryList.npy')
synapses = np.load('WList.npy')
#%%
nIter = 10
M = 15
N= 200
modes = ['Full','Feedforward','None']
xTarget = np.empty((nIter*M,N))
x = {}
for mode in modes: x[mode] = np.empty((nIter*M,N))
#%%
from matplotlib import cm
ii = np.random.randint(0,nIter)
k = np.random.randint(0,M)
ind = M*ii+k
x_target = memorys[ii][:,k]
x_t = results[ind][modes[0]]['y']
t = results[ind][modes[0]]['t']
for i in np.random.randint(0,N,20):
    p = x_target[i]
    plt.plot(t,x_t[i,:],c=cm.hsv(p/(2*np.pi)))
#%%
for ii in range(nIter):
    for k in range(M):
        ind = M*ii+k
        xTarget[ind] = memorys[ii][:,k]
        for mode in modes:
            sol = results[ind][mode]
            x[mode][ind] = sol['y'][:,-1]
#%%
# # print(xTarget[0].shape)
# ii=2
# k=11

# print(np.where(np.isnan(xTarget[41])))
#%%
dx = {}
R = {}
bias= {}
var = {}
error = {}
for mode in modes:
    xx = np.mod(x[mode],np.pi*2)
    dx[mode] = xx - xTarget
    dx[mode] = dx[mode].flatten()
    dx[mode] = wrap(dx[mode])
#%%
for mode in modes:
    R[mode] = circR(dx[mode])
    bias[mode] = np.angle(R[mode])
    var[mode] = 1 - abs(R[mode])
    error[mode] = var[mode] + 2*abs(R[mode])*(np.sin(bias[mode]/2)**2)
print(bias)
print(var)
print(error)
#%%
for mode in modes:
    h = plt.hist(dx[mode],100,(-np.pi,np.pi),density=True,histtype='step',label=mode)
    plt.legend()
        


