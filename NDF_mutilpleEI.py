#%%
import numpy
from matplotlib import pyplot
from scipy.integrate import odeint
sig = lambda x: 1/(1+numpy.exp(-x))

#%% parameter
# Hopfield
N=100 #number of units
s=5 #number of traces
nPerturb = 2 # number of units fliped in cue
# NDF
tEE = 100 # time constant of EE synapses
tEI = 10 # time constant of EI synapses
tIE = 25 # time constant of IE synapses
tII = 10 # time constant of II synapses
tO = 5 # time constant of external input
J = 100 # synaptic strength
JEE = J
MEE = JEE*numpy.eye(N) # initate EE synapses
JEI = J
MEI = JEI*numpy.eye(N)
JIE = J
MIE = JIE*numpy.eye(N)
JII = J
MII = JII*numpy.eye(N)
# JO  = 1e9
# Hopfield
x = numpy.random.randint(0,2,(N,s))
temp = x*2-1
cov_ = temp @ temp.T - s*numpy.eye(N) # correlation synapse
MEE += cov_*JEE
# MEE[cov_>0] += cov_[cov_>0]*JEE
# MEI[cov_<0] -= cov_[cov_<0]*JEI
# MIE[cov_<0] -= cov_[cov_<0]*JIE
# MII[cov_>0] += cov_[cov_>0]*JII
#
# stim = lambda t:int(t>0 and t<=1)
# mainode
def NDF(y,t):
    sEE = y[0:N]
    sEI = y[N:N*2]
    sIE = y[N*2:N*3]
    sII = y[N*3:N*4]
    E = sig(MEE@sEE-MEI@sEI) 
    I = sig(MIE@sIE-MII@sII)
    dEE = (-sEE + E)/tEE
    dEI = (-sEI + I)/tEI
    dIE = (-sIE + E)/tIE
    dII = (-sII + I)/tII
    return numpy.concatenate((dEE,dEI,dIE,dII))

#%% initial values
J0 = 0.2
E0 = x[:,0]-0.5
E0[:nPerturb] = -E0[:nPerturb]
pyplot.bar(range(N),E0,label='altered')
pyplot.bar(range(N),x[:,0]-0.5,label='original')
pyplot.xlabel('index')
pyplot.ylabel('cue')
pyplot.legend()
I0 = numpy.zeros(N)
sEE0 = 0.5 + J0 * E0 / tEE
sEI0 = 0.5 + J0 * I0 / tEI
sIE0 = 0.5 + J0 * E0 / tIE
sII0 = 0.5 + J0 * I0 / tII
y0 = numpy.concatenate((sEE0,sEI0,sIE0,sII0))
t = numpy.arange(1000)
#%% solve
y = odeint(NDF,y0,t)
# sEE = y[:,0:N]
# sEI = y[:,N:N*2]
# sIE = y[:,N*2:N*3]
# sII = y[:,N*3:N*4]
yt = y[-1,:]
sEEt = yt[0:N]
sEIt = yt[N:N*2]
sIEt = yt[N*2:N*3]
sIIt = yt[N*3:N*4]
Et = sig(MEE@sEEt-MEI@sEIt)
pyplot.bar(range(N),(Et-0.5))
pyplot.xlabel('index')
pyplot.ylabel('final - 0.5')
error = Et - x[:,0]
print(numpy.where(error!=0))
print(error[error!=0])
#%%
sEEc1 = y[:,3]
pyplot.plot(sEEc1)
pyplot.xlabel('time')
pyplot.ylabel('sEE[3]')
sEEc1[0]
    


