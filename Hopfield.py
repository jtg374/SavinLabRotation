#%%
import numpy
from matplotlib import pyplot


#%% Parameters 
N=1000 #number of units
s=10 #number of traces
T=20 # total timesteps

#%% Initialization
# generate random activity trace
x = numpy.random.randint(0,2,(N,s))

# generate symmetric connection based on random traces
temp = x*2-1
M = temp @ temp.T
for i in range(N):
    M[i,i]=0

#%% Retrival
s_cue = numpy.random.choice(s)
x_curr = x[:,s_cue]
x_bar = []
for t in range(T):
    x_bar.append(x_curr)
    order = numpy.random.permutation(N)
    for i in order:
        x_curr[i] = numpy.dot(M[i],x_curr)>0
    
x_bar = numpy.array(x_bar)

#%% output result
for i in [0,10,20,30,40]:
    pyplot.plot(x_bar[:,i])
pyplot.xlabel('time')
pyplot.ylabel('some units')
pyplot.show()
#%%
x_std = x_bar.std(0)
pyplot.plot(x_std)
pyplot.xlabel('i')
pyplot.ylabel('std(x_i)')