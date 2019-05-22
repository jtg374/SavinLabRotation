import numpy as np
def HMC(U, grad_U,step,L,current_q):
    q = current_q
    p = np.random.normal(0,1,len(q)) # independent Standard Multivariate Guassian
    current_p = p

    # Make a half step for momentum p at the beginning
    p += -step * grad_U(q) / 2

    # Alternate full steps for position q and momentum p
    for t in range(L):
        # update position
        q += step*p
        # update momentum (except for last step)
        if not t==(L-1):
            p += -step * grad_U(q)
        else:
            p += -step * grad_U(q) / 2
    
    # Negate memontum at end of trajectory to make the proposal symmetric
    # p=-p

    # Evaluate potential and kinetic energy 
    current_H = U(current_q) + np.sum(current_p**2)/2
    proposed_H = U(q) + np.sum(p**2) / 2

    # Accept or reject
    if np.random.uniform() < np.exp(current_H - proposed_H):
        return q # accept
    else:
        return current_q # reject
