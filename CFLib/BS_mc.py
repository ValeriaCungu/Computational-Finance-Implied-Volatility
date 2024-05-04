from math import *
import numpy as np

def BS_trj(Obj, nt, So, T, J, sigma):

    '''
    Generates J trajectories according to the Black+Scholes model
    Each trajectory is made up of nt equally spaced steps.

    The output matrix will have the geometry S[nt+1][J],
    For each trajectory S[0] will hold the initial value.
    '''

    DT = T/nt
    S  = np.ndarray(shape = ( nt+1, J), dtype=np.double)

    #The +1 in nt+1 accounts for the initial asset price at time zero, 
    #The shape argument specifies the dimensions of the S array. Each row in this array represents a point in time, 
    #while each column represents a different trajectory or simulation of the asset price over time.
    #nt rows where nt is the single periods for example 1 month, 2 month etc
    #J are the columns, this means the trajectories 

    X  = Obj.normal( -.5*sigma*sigma*DT, sigma*sqrt(DT), (nt,J))

    S[0] = So
    #la prima riga che è il primo elemento di ogni simulazione è posto come il valore che S ha al tempo = S0

    for n in range(nt):
        # for j in range(0, J): 
        #   S[n+1,j] = S[n,j] * exp( X[n,j] )
        S[n+1] = S[n] * np.exp(X[n])

    return S

Obj = np.random.RandomState(1)
nt = 12
So = 1
T = 1
J = 10000
sigma = 0.2
print(BS_trj(Obj, nt, So, T, J, sigma))




