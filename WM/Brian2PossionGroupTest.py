#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
from brian2 import *
G = NeuronGroup(3,'r: Hz',threshold='rand()<r*dt',name='output')
G.r = np.zeros(3)*Hz
Input = PoissonGroup(3,np.array([1,2,3])*Hz,name='input')
S = Synapses(Input,G,on_pre='r_post += 0.1*Hz')
S.connect(j='i')

M = StateMonitor(G,'r',record=True)
run(1000*ms)

#%%
plt.plot(M.t/ms,M.r[2])

#%%
