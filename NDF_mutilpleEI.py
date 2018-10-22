#%%
import numpy
from matplotlib import pyplot
from scipy.integrate import odeint
sig = lambda x: 1/(1+numpy.exp(-x))

#%% parameter
# Hopfield
N=100 #number of units
s=5 #number of traces
# NDF
tEE = 100
tEI = 10
tIE = 25
tII = 10
tO = 5
JEE = 100
MEE = JEE*numpy.eye(N)*1
JEI = 100
MEI = JEI*numpy.eye(N)*1
JIE = 100
MIE = JIE*numpy.eye(N)*1
JII = 100
MII = JII*numpy.eye(N)*1
JO  = 1e9
perturbation_cue = 3 # number of units to change in cue
# Hopfield
x = numpy.random.randint(0,2,(N,s))
temp = x*2-1
cov_ = temp @ temp.T - s*numpy.eye(N)
MEE[cov_>0] += cov_[cov_>0]*JEE*10
MEI[cov_<0] -= cov_[cov_<0]*JEI*10
# MIE[cov_<0] -= cov_[cov_<0]*JIE*10
# MII[cov_>0] += cov_[cov_>0]*JII*10
#
# stim = lambda t:int(t>0 and t<=1)
# mainode
def NDF(y,t):
    sEE = y[0:N]
    sEI = y[N:N*2]
    sIE = y[N*2:N*3]
    sII = y[N*3:N*4]
    iE  = y[N*4:N*5]
    E = sig(MEE@sEE-MEI@sEI+iE) 
    I = sig(MIE@sIE-MII@sII+iE) ###
    dEE = (-sEE + E)/tEE
    dIE = (-sIE + E)/tIE
    dEI = (-sEI + I)/tEI
    dII = (-sII + I)/tII
    diE = (-iE  + JO*stim(t)*stim0)/tO
    return numpy.concatenate((dEE,dIE,dEI,dII,diE))

#%% initial values (transient stim)
J0 = 0.2
E0 = numpy.zeros(N)
stim = lambda t:int(t>0 and t<=1)
stim0 = x[:,0].copy()
stim0[:perturbation_cue] = 1-stim0[:perturbation_cue]
pyplot.bar(range(N),x[:,0]-0.5)
pyplot.bar(range(N),stim0-0.5)
pyplot.xlabel('index')
pyplot.ylabel('cue - 0.5')
I0 = numpy.zeros(N)
sEE0 = 0.5 + J0 * E0 * tEE / tEE
sEI0 = 0.5 + J0 * I0 * tEE / tEI
sIE0 = 0.5 + J0 * E0 * tEE / tIE
sII0 = 0.5 + J0 * I0 * tEE / tII
iE0  = numpy.zeros(N)
y0 = numpy.concatenate((sEE0,sEI0,sIE0,sII0,iE0))
t = numpy.arange(1000)
#%% initial values (initial state)
J0 = 0.2
E0 = x[:,0].copy()
stim = lambda t:int(t>0 and t<=1)
stim0 = numpy.zeros(N)
E0[:perturbation_cue] = 1-E0[:perturbation_cue]
pyplot.bar(range(N),x[:,0]-0.5)
pyplot.bar(range(N),E0-0.5)
pyplot.xlabel('index')
pyplot.ylabel('cue - 0.5')
I0 = E0
sEE0 = 0.5 + J0 * E0 * tEE / tEE
sEI0 = 0.5 + J0 * I0 * tEE / tEI
sIE0 = 0.5 + J0 * E0 * tEE / tIE
sII0 = 0.5 + J0 * I0 * tEE / tII
iE0  = numpy.zeros(N)
y0 = numpy.concatenate((sEE0,sEI0,sIE0,sII0,iE0))
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
    


