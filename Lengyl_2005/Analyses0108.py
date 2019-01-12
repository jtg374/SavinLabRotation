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
time = '01070128'
memorys_import = np.load('xMemory'+time+'.npy')
results_import = np.load('xFinal'+time+'.npy')
#%%
k_prior = 0.5
k_cue = 10
#%%
memorys = np.mod(memorys_import,np.pi*2)
results = np.mod(results_import,np.pi*2)
cues = memorys + np.random.vonmises(0,k_cue,memorys.shape)
#%%
modes = ['Full','Feedforward','None']
#%%
x = {}
x[modes[0]] = results
x[modes[1]] = ff(cues,k_prior/k_cue)
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
time = '01061300'
memorys_import = np.load('xMemory'+time+'.npy')
results_import = np.load('xFinal'+time+'.npy')
memorys_import = np.mod(memorys_import,np.pi*2)
results_import = np.mod(results_import,np.pi*2)
memorys = np.append(memorys,memorys_import)
results = np.append(results,results_import)
cues = memorys + np.random.vonmises(0,k_cue,memorys.shape)


#%%
x[modes[0]] = results
x[modes[1]] = ff(cues,k_prior/k_cue)
x[modes[2]] = np.zeros_like(results)

#%%
dx = {}
for mode in modes:
    dx[mode] = x[mode]-memorys
    dx[mode] = dx[mode].flatten()
    dx[mode] = wrap(dx[mode])

h = {}
for mode in modes:
    h[mode] = plt.hist(dx[mode],100,(-np.pi,np.pi),density=True,histtype='step',label=mode)
    plt.legend()

#%%
time = '01070114'
memorys_import = np.load('xMemory'+time+'.npy')
results_import = np.load('xFinal'+time+'.npy')
memorys_import = np.mod(memorys_import,np.pi*2)
results_import = np.mod(results_import,np.pi*2)
memorys = np.append(memorys,memorys_import)
results = np.append(results,results_import)
cues = memorys + np.random.vonmises(0,k_cue,memorys.shape)

x[modes[0]] = results
x[modes[1]] = ff(cues,k_prior/k_cue)
x[modes[2]] = np.zeros_like(results)

dx = {}
for mode in modes:
    dx[mode] = x[mode]-memorys
    dx[mode] = dx[mode].flatten()
    dx[mode] = wrap(dx[mode])

h = {}
for mode in modes:
    h[mode] = plt.hist(dx[mode],100,(-np.pi,np.pi),density=True,histtype='step',label=mode)
    plt.legend()


#%%
ax = plt.subplot(111,projection='polar')
for mode in modes:
    r,a = np.histogram(dx[mode],400,(-np.pi,np.pi))
    r = np.append(r,r[0])
    line = ax.plot(a,r,drawstyle='steps-pre',label=mode)
ll=plt.legend()


#%%
