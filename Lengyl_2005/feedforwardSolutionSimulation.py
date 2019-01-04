#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
k_prior = 5
kratios = np.logspace(-2,8,31,base=2)
# print(kratios)
#%%
NN = int(1e5)
bias = np.zeros_like(kratios)
lvar = np.zeros_like(kratios)
ldsp = np.zeros_like(kratios)
cvar = np.zeros_like(kratios)
cdsp = np.zeros_like(kratios)
#%%
from matplotlib import cm
for ind,k in enumerate(kratios):
    # print(ind)
    x = np.random.vonmises(0,k_prior,NN)
    noise = np.random.vonmises(0,k*k_prior,NN)
    x_tilde = x + noise
    Z = np.exp(1j*x_tilde) + 1/k
    x_bar = np.angle(Z)
    error = x_bar - x
    error = np.mod(error+np.pi,np.pi*2)-np.pi
    # h=plt.hist(error,100,(-np.pi,np.pi),color=cm.YlGn(ind/21),histtype='step')
    bias[ind] = np.mean(error)
    lvar[ind] = np.var(error)
    ldsp[ind] = lvar[ind] + bias[ind]**2
    vR = np.exp(1j*error)
    R = abs(np.mean(vR))
    cvar[ind] = 1 - R
    cdsp[ind] = cvar[ind] + 2*R*(np.sin(bias[ind]/2)**2)
#%%
# plt.plot(kratios,bias)
#%%
# plt.plot(kratios,bias**2)
plt.semilogx(kratios,lvar)
plt.semilogx(kratios,ldsp)
plt.xlabel(r'k_{cue}/k_{prior}')
plt.ylabel('linear MSE')
#%%
# plt.plot(kratios,bias**2)
plt.semilogx(kratios,cvar)
plt.semilogx(kratios,cdsp)
plt.xlabel(r'k_{cue}/k_{prior}')
plt.ylabel('circular MSE')



