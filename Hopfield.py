#%%
import numpy
from matplotlib import pyplot


#%% Parameters 
N=1000 #number of units
s=10 #number of traces
T=20 # total timesteps
perturbation_cue = 0 # number of units to change in cue
retrival_is_deterministic = False
sig = lambda x: 1/(1+numpy.exp(-x))

#%% Initialization
# generate random activity trace
x = numpy.random.randint(0,2,(N,s))

# generate symmetric connection based on random traces
temp = x*2-1
M = temp @ temp.T
for i in range(N):
    M[i,i]=0

#%% Retrival
s_cue = 0 # numpy.random.choice(s)
x_curr = x[:,s_cue].copy()
i_flip = [0] # numpy.random.choice(N,perturbation_cue,replace=False)
x_curr[i_flip] = 1 - x_curr[i_flip]
print(x_curr[:10])
x_bar = []
for t in range(T):
    x_bar.append(x_curr)
    order = numpy.random.permutation(N)
    for i in order:
        I_syn = numpy.dot(M[i],x_curr)
        if retrival_is_deterministic:   
            fire = I_syn>0
        else:
            fire = sig(I_syn) > numpy.random.rand()
        x_curr[i] = fire
    
x_bar = numpy.array(x_bar)

#%% output result
for i in i_flip:
    pyplot.plot(x_bar[:,i])
pyplot.xlabel('time')
pyplot.ylabel('some units')
pyplot.show()
#%%
x_std = x_bar.std(0)
pyplot.plot(x_std)
pyplot.xlabel('i')
pyplot.ylabel('std(x_i)')

#%%
print(x_bar[0:3,0])
