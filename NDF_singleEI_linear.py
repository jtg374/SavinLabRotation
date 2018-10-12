#%%
import numpy
from matplotlib import pyplot
from scipy.integrate import odeint
# sig = lambda x: 1/(1+numpy.exp(-x))

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
def NDF(y,t):
    sEE,sEI,sIE,sII,iE = y
    E = (JEE*sEE-JEI*sEI+iE) 
    I = (JIE*sIE-JII*sII)
    dEE = (-sEE + E)/tEE
    dIE = (-sIE + E)/tIE
    dEI = (-sEI + I)/tEI
    dII = (-sII + I)/tII
    diE = (-iE  + JO*stim(t))/tO
    return [dEE,dEI,dIE,dII,diE]
#%%
y0 = [0,0,0,0,0]
t = numpy.arange(1000)
for JO in [-2,2,3,4,5,9,12]:
    y = odeint(NDF,y0,t)
    sEE,sEI,iE = y[:,0],y[:,1],y[:,4]
    E = (JEE*sEE-JEI*sEI+iE) 
    pyplot.plot(E) 
    # E is not graded while sEE is, I dont understand

#%%
iE = y[:,4]
pyplot.plot(iE)
#%%
sEE = y[:,0]
pyplot.plot(sEE)