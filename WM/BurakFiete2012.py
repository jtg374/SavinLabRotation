#%%
import numpy as np
import matplotlib.pyplot as plt
import brian2 as b2
#%%
from brian2 import ms,Hz,NeuronGroup,Synapses
from numpy import pi,sin,cos,exp
#%%
b2.start_scope()
#%%
# Parameters
N = 1024 # number of Neurons
b = -2 # background inhibition
k1 = 1 # concentration parameter for recurrent excitation
k2 = .3 # recurrent inhibition

