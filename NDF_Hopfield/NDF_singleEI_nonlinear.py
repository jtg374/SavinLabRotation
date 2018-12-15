#%%
import numpy
from matplotlib import pyplot
from scipy.integrate import odeint
sig = lambda x: 1/(1+numpy.exp(-x))

#%% parameter
tEE = 100
tEI = 10
tIE = 25
tII = 10
tO = 5
JEE = 100
JEI = 100
JIE = 100
JII = 100
JO  = 50
stim = lambda t:int(t>0 and t<=1)
# main ode
def NDF(t,y):
    sEE,sEI,sIE,sII,iE = y
    E = sig(JEE*sEE-JEI*sEI+iE) 
    # stim here makes E less like integrator under long stim
    I = sig(JIE*sIE-JII*sII)
    dEE = (-sEE + E)/tEE
    dIE = (-sIE + E)/tIE
    dEI = (-sEI + I)/tEI
    dII = (-sII + I)/tII
    diE = (-iE  + JO*stim(t))/tO
    return [dEE,dEI,dIE,dII,diE]
#%% stim by external
y0 = [.5,.5,.5,.5,0]
t = numpy.arange(1000)
for JO in [-200,200,2e3,2e4,2e5,2e9,2e12]:
    y = odeint(NDF,y0,t,tfirst=True)
    sEE,sEI,sIE,sII,iE = y[:,0],y[:,1],y[:,2],y[:,3],y[:,4]
    E = sig(JEE*sEE-JEI*sEI+iE) 
    pyplot.plot(E)
#%% stim by initial value
JO = 0
t = numpy.arange(1000)
for E0 in [-0.005,0.005,0.01,0.02,0.04]:
    y0 = [E0+0.5,E0+0.5,.5,.5,0]
    y = odeint(NDF,y0,t,tfirst=True)
    sEE,sEI,sIE,sII,iE = y[:,0],y[:,1],y[:,2],y[:,3],y[:,4]
    E = sig(JEE*sEE-JEI*sEI+iE) 
    pyplot.plot(E)

#%%
iE = y[:,4]
pyplot.plot(iE)
#%%
sEE = y[:,0]
pyplot.plot(sEE)