#!/usr/bin/env python3

from math import *
import matplotlib.pyplot as plt
import numpy as np
from time import time
try:
    from CIR import CIR,  cir_evol, QT_cir_evol
except ModuleNotFoundError:
    from CFLib.CIR import CIR,  cir_evol, QT_cir_evol
# -----------------------------------------------------

def __mc_heston__( rand, So, vol, intVol, cir, rho, Dt, N  ):

    '''
    @parms So    : initial value
    @parms intVol: volatility integral trajectory
    @parms cir   : CIR object
    @parms rho   : correlation between vol and underlying innovations
    @parms Dt    : tenor of the underlying trajectory
                   must agree with the nodes of the volatility trajectory
    @parms N     : number of underlying trajectories
    '''

    # length of the volatility trajectory
    # (including initial point)
    L   = len(intVol)
    th  = cir.theta
    k   = cir.kappa
    eta = cir.sigma
    nu  = vol
    I   = intVol

    # underlying trajectorie
    S  = np.ndarray(shape = (L, N), dtype=np.double ) # S[N, L] in fortran matrix notation

    xi = rand.normal( loc = 0.0, scale = 1.0, size=(L-1, N))

    # prime with So the starting value of each trajectory
    S[0] = So

    for n in range(1,L):
        DI   = I[n] - I[n-1]
        X    = -.5 * DI + (rho/eta)*( nu[n] - nu[n-1] - k*( th*Dt - DI) ) + sqrt((1. - rho*rho)*DI)*xi[n-1]
        S[n] = S[n-1]*np.exp(X)

    return S

# ----------------------------------------------------

def mc_heston(rand, So, vol, intVol, cir, rho, Dt, N  ):
    return __mc_heston__(rand, So, vol, intVol, cir, rho, Dt, N  )

def heston_trj( rand_1
               ,rand_2
               ,So
               ,lmbda  # kappa
               ,eta    # sigma
               ,nubar  # theta
               ,nu_o   # ro
               ,rho    # correlation
               ,Yrs    # Length of the trajectory in years
               ,dt     # step per vol inegration
               ,Nt     # number of steps in the S trajectory
               ,NV     # number of vol trajectories
               ,NS     # number of S trajectory per vol trajectory
              ):
    cir = CIR(kappa=lmbda, sigma=eta, theta=nubar, ro = nu_o)

    nCir      = int(Yrs/dt) 
    dt        = Yrs/nCir
    Dt        = Yrs/Nt
    blockSize = (2 << 8 )
    if NV > blockSize : 
        NB = NV//blockSize
    else:
        NB = 1
        blockSize = NV

    rem = NV - NB*blockSize

    S   = np.zeros(shape=(NV, Nt+1, NS), dtype=np.float64) 
    for nb in range(NB):
        vol, Ivol = QT_cir_evol( rand_1, cir, nCir, dt, Nt, Dt, blockSize)
        Ivol = Ivol.transpose()
        vol = vol.transpose()
        for n in range(blockSize):
            s = __mc_heston__( rand_2, So, vol[n], Ivol[n], cir, rho, Dt, NS )
            S[n + nb*blockSize] = s
    if rem > 0:
        vol, Ivol = QT_cir_evol( rand_1, cir, nCir, dt, Nt, Dt, rem)
        Ivol = Ivol.transpose()
        vol = vol.transpose()
        for n in range(rem):
            s = __mc_heston__( rand_2, So, vol[n], Ivol[n], cir, rho, Dt, NS )
            S[NB*blockSize+n] = s

    return S

