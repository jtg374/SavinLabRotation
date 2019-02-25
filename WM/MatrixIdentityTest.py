#%%
import numpy as np
#%% 
# Parameters
N1 = 2**10
N2 = 2**12
gFF = 100.
gFB = 50.
p = 0.2
#%%
# connection
C = np.random.uniform(0,1,(N2,N1))<p
N_exc_s = np.sum(C,0)
N_exc_r = np.sum(C,1)
#%%
# Feedforward weight
k1 = 2100
wFF = -k1/(N2) + k1*C/N_exc_s[None,:] 
# and Feedback Weight
k2 = 200
wFB = -k2/N1 + k2*C.T/N_exc_r[None,:]
#%%
W = wFF@wFB

#%%
W.shape

#%%
u,s,vh = np.linalg.svd(W)
print(s)

#%%
import matplotlib.pyplot as plt
plt.matshow(W)
plt.colorbar()

#%%
np.sum(wFB,0)

#%%
