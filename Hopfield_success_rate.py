#%%
import numpy
from matplotlib import pyplot


#%% Parameters 
N=100 #number of units
s=10 #number of traces
perturbation_cue = 0 # number of units to change in cue
retrival_is_deterministic = True
sig = lambda x: 1/(1+numpy.exp(-x))
N_run=500

#%% main loop
freq = numpy.zeros(N+1)
for i_run in range(N_run):
    #%% Initialization
    # generate random activity trace
    x = numpy.random.randint(0,2,(N,s))

    # generate symmetric connection based on random traces
    temp = x*2-1
    M = temp @ temp.T
    for i in range(N):
        M[i,i]=0

    #%% Retrival
    # retrival cue
    s_cue = 0 # numpy.random.choice(s)
    x_curr = x[:,s_cue].copy()
    i_flip =  numpy.random.choice(N,perturbation_cue,replace=False)
    x_curr[i_flip] = 1 - x_curr[i_flip]
    # retrival, compare subsequent trace
    x_last = x_curr.copy()
    t=0
    while (not numpy.array_equal(x_last,x_curr)) and t<100: # break if trace doesn't change
        x_last = x_curr.copy()
        order = numpy.random.permutation(N)
        for i in order:
            I_syn = numpy.dot(M[i],x_curr)
            if retrival_is_deterministic:   
                fire = I_syn>0
            else:
                fire = sig(I_syn) > numpy.random.rand()
            x_curr[i] = fire
        t+=1
    error = numpy.sum(numpy.abs(x_curr-x[:,s_cue]))
    freq[error]+=1

freq/=N_run

#%%
print(freq[0])
pyplot.bar(range(N+1),freq)

