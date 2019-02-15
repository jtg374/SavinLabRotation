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
C = np.random.uniform(0,1,(N1,N2))<p
#%%
# Feedforward weight
wFF = np.zeros_like(C,dtype=float)
for i in range(N1):
    wFF[i] = gFF*C[i] / np.sum(C[i])
wFF = wFF - gFF/N2
#%%
# and Feedback Weight
wFB = np.zeros_like(C.T,dtype=float)
for i in range(N2):
    wFB[i] = C.T[i] / np.sum(C.T[i])
wFB = wFB*gFB - gFB/N1
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
