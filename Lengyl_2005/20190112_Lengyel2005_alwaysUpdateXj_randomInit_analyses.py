#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
plt.style.use('default')
plt.style.use('seaborn-talk')

#%%
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
k_prior = 0.5
k_cue = 10
nIter=10
N=200
M=10
#%%
memorys = np.empty((nIter,N,M))
cues = np.empty((nIter,N,M))
results = np.empty((nIter,N,M))
for iIter in range(nIter):
    filename = 'Data/Lengyel2005_alwaysUpdateXj_Loop/'+'Lengyel2005_alwaysUpdateXj_iter%02d.npz'%(iIter)
    D = np.load(filename)
    # D.files
    memorys[iIter] = D['xMemory']
    cues[iIter] = D['xCues']
    results[iIter] = D['xRecalled'][:,-1,:]

#%%
memorys = np.mod(memorys,np.pi*2)
results = np.mod(results,np.pi*2)
cues = np.mod(cues,np.pi*2)
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
    h[mode] = plt.hist(dx[mode],100,(-np.pi,np.pi),density=True,histtype='step',label=mode)[0]
plt.xlabel('error')
plt.ylabel('count')
plt.legend()

#%%
ax = plt.subplot(111,projection='polar')
for mode in modes:
    r,a = np.histogram(dx[mode],400,(-np.pi,np.pi))
    r = np.append(r,r[0])
    line = ax.plot(a,r,drawstyle='steps-pre',label=mode)
ll=plt.legend()
#%%
R = {}
bias= {}
var = {}
lsd = {}
error = {}
for mode in modes:
    lsd[mode] = np.std(dx[mode])
    R[mode] = circR(dx[mode])
    bias[mode] = np.angle(R[mode])
    var[mode] = 1 - abs(R[mode])
    error[mode] = var[mode] + 2*abs(R[mode])*(np.sin(bias[mode]/2)**2)
print('mean bias: \n',bias)
print('linear root mean squared error: \n',lsd)
print('circular variance: \n',var)
print('circular dispersion: \n',error)
#%%
x_t = D['xRecalled'][:,:,0]
t = D['t_eval']
inds = np.where(t>125)
plt.plot(t[inds],x_t[0,inds].flatten())




#%%
