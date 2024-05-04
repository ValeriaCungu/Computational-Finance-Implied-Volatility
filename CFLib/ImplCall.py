import sys
from sys import stdout as cout
from scipy.stats import norm
import numpy as np
from math import *

def cn_put( T, sigma, kT):
    s    = sigma*sqrt(T)
    d    = ( np.log(kT) + .5*s*s)/s
    return norm.cdf(d)
# ------------------------------------

def an_put( T, sigma, kT):
    s    = sigma*sqrt(T)
    d    = ( np.log(kT) + .5*s*s)/s
    return norm.cdf(d-s)
def FwEuroPut(T, vSigma, vKt):
    return ( vKt* cn_put( T, vSigma, vKt) - an_put( T, vSigma, vKt) )

def FwEuroCall(T, vSigma, vKt):
    return FwEuroPut(T, vSigma, vKt) + 1. - vKt

def euro_put(So, r, q, T, sigma, k):
    kT   = exp((q-r)*T)*k/So
    return So*exp(-q*T) * FwEuroPut( T, sigma, kT)
# -----------------------

def euro_call(So, r, q, T, sigma, k):
    kT   = exp((q-r)*T)*k/So
    # ( exp(-rT)*K* cn_put( T, vSigma, vKt) - So*exp(-qT)*an_put( T, vSigma, vKt) )
    return So*exp(-q*T) * FwEuroCall( T, sigma, kT)
# -----------------------

def impVolFromFwCall(vPrice, T, vKt):

    scalar = isinstance(vKt, float)
    if scalar: vKt = np.array([vKt])

    vSl = np.zeros(vKt.shape[0])
    vPl = np.maximum(vKt - 1., 0.0)

    vSh = np.ones(vKt.shape[0])
    while True:
        vPh = FwEuroCall(T, vSh, vKt)
        if ( vPh > vPrice).all(): break
        vSh = 2*vSh

    # d = vSh-vSl
    # d/2^N < eps
    # d < eps* 2^N
    # N > log(d/eps)/log(2)
    eps = 1.e-08
    d   = vSh[0]-vSl[0]
    N   = 2+int(log(d/eps)/log(2))

    for n in range(N):
        vSm  = .5*(vSh + vSl)
        vPm  = FwEuroCall(T, vSm, vKt)
        mask = vPm > vPrice
        vSh[mask] = vSm[mask]
        vSl[~mask] = vSm[~mask]

    
    if scalar: return .5*(vSh + vSl)[0]
    return .5*(vSh + vSl)