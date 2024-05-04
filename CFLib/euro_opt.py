#!/usr/bin/env python3

import sys
from sys import stdout as cout
from scipy.stats import norm
import numpy as np
from math import *
try:
    from .config import get_input_parms
except ImportError:
    from config import get_input_parms

def cn_put( T, sigma, kT):
    s    = sigma*sqrt(T)
    d    = ( np.log(kT) + .5*s*s)/s
    return norm.cdf(d)
# ------------------------------------

def an_put( T, sigma, kT):
    s    = sigma*sqrt(T)
    d    = ( np.log(kT) + .5*s*s)/s
    return norm.cdf(d-s)
# ------------------------------------
#
#def cn_put( T, sigma, kT):
#    if kT == 0.0: return 0.0
#    s    = sigma*sqrt(T)
#    if s < 1.e-08:
#        if kT > 1.0: return 1.
#        else       : return 0.
#    d    = ( log(kT) + .5*s*s)/s
#    return norm.cdf(d)
# ------------------------------------

#def an_put( T, sigma, kT):
#    if kT == 0.0: return 1.0
#    s    = sigma*sqrt(T)
#    if s < 1.e-08:
#        if kT > 1.0: return 1.0 
#        else       : return 0.
#    d    = ( log(kT) + .5*s*s)/s
#    return norm.cdf(d-s)
# ------------------------------------

'''
    PUT = exp(-rT)Em[ (K - S(T))^+]
        where
    S(T) = So exp( (r-q)*T)*M
        let
    Fw(T) = So exp( (r-q)*T)
    kT    = K/Fw
        then
    PUT = So exp(-qT) Em[ (kT - M)^+]
        = So exp(-qT) FwEuroPut( T, sigma, kT)
'''
#def FwEuroPut(T, sigma, kT):
#    return ( kT* cn_put( T, sigma, kT) - an_put( T, sigma, kT) )

#def FwEuroCall(T, sigma, kT):
#    return FwEuroPut(T, sigma, kT) + 1. - kT

def FwEuroPut(T, vSigma, vKt):
    return ( vKt* cn_put( T, vSigma, vKt) - an_put( T, vSigma, vKt) )

def FwEuroCall(T, sigma, vkT):
    return FwEuroPut(T, sigma, vkT) + 1. - vkT

def euro_put(So, r, q, T, sigma, k):
    kT   = exp((q-r)*T)*k/So
    return So*exp(-q*T) * FwEuroPut( T, sigma, kT)
# -----------------------

def euro_call(So, r, q, T, sigma, k):
    kT   = exp((q-r)*T)*k/So
    # ( exp(-rT)*K* cn_put( T, vSigma, vKt) - So*exp(-qT)*an_put( T, vSigma, vKt) )
    return So*exp(-q*T) * FwEuroCall( T, sigma, kT)
# -----------------------

def impVolFromFwPut(vPrice, T, vKt):

    scalar = isinstance(vKt, float)
    if scalar: vKt = np.array([vKt])

    vSl = np.zeros(vKt.shape[0])
    vPl = np.maximum(vKt - 1., 0.0)

    vSh = np.ones(vKt.shape[0])
    while True:
        vPh = FwEuroPut(T, vSh, vKt)
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
        vPm  = FwEuroPut(T, vSm, vKt)
        mask = vPm > vPrice
        vSh[mask] = vSm[mask]
        vSl[~mask] = vSm[~mask]

    
    if scalar: return .5*(vSh + vSl)[0]
    return .5*(vSh + vSl)

# --------------------------------------------

def vanilla_options( **keywrds):

    So     = keywrds["S"]
    k      = keywrds["k"]
    r      = keywrds["r"]
    q      = keywrds.get("q", 0.0)
    T      = keywrds["T"]
    sigma  = keywrds["sigma"]
    fp     = keywrds.get("fp", None)

    if not fp is None:
        fp.write("@ %-24s %8.4f\n" %("So", So))
        fp.write("@ %-24s %8.4f\n" %("k", k))
        fp.write("@ %-24s %8.4f\n" %("T", T))
        fp.write("@ %-24s %8.4f\n" %("r", r))
        fp.write("@ %-24s %8.4f\n" %("q", q))
        fp.write("@ %-24s %8.4f\n" %("sigma", sigma))

    kT   = exp((q-r)*T)*k/So
    cnP  = k*exp(-r*T)*cn_put ( T, sigma, kT)
    anP  = So*exp(-q*T)*an_put ( T, sigma, kT)
    put  = euro_put ( So, r, q, T, sigma, k)
    call = euro_call( So, r, q, T, sigma, k)

    return {"put": put, "call": call, "anP": anP, "cnP": cnP}
# --------------------------
def usage():
    print("Computes the value of european Call/Put options")
    print("and put-cash or nothing and put asset or nothing")
    print("Usage: $> ./euro_opt.py [options]")
    print("Options:")
    print("    %-24s: this output" %("--help"))
    print("    %-24s: output file" %("-out outputFile"))
    print("    %-24s: initial value of the underlying, defaults to 1.0" %("-s So"))
    print("    %-24s: option strike, defaults to 1.0" %("-k strike"))
    print("    %-24s: option strike, defaults to .40" %("-v volatility"))
    print("    %-24s: option maturity, defaults to 1.0" %("-T maturity"))
    print("    %-24s: interest rate, defaults to 0.0" %("-r ir"))
# ----------------------------------

def run(args):

    output    = None
    So     = 1.0
    k      = 1.0
    T      = 1.0
    r      = 0.0
    q      = 0.0
    Sigma  = .2347
    inpts  = get_input_parms(args)

    if "help" in inpts:
        usage()
        return

    try:
        output = inpts["out"]
        fp     = open(output, "w")
    except KeyError:
        fp     = cout

    try: So = float( inpts["So"] )
    except KeyError: pass

    try: k = float( inpts["k"] )
    except KeyError: pass

    try: T = float( inpts["T"] )
    except KeyError: pass

    try: r = float( inpts["r"] )
    except KeyError: pass

    try: Sigma = float( inpts["v"] )
    except KeyError: pass

    res = vanilla_options(fp=fp, T=T, r =r, q=q, sigma=Sigma, k = k, S = So)
    kT = exp((q-r)*T)*k/So
    fwPut = FwEuroPut(T, Sigma, kT)
    impVol = impVolFromFwPut(fwPut, T, kT)

    fp.write("@ Put %14.10f,  Call %14.10f,  Pcn %14.10f,  Pan %14.10f  ImpVol: %10.6f\n" %(res["put"], res["call"], res["cnP"], res["anP"], impVol))

    if output != None: fp.close()
# --------------------------

if __name__ == "__main__":
    run(sys.argv)
