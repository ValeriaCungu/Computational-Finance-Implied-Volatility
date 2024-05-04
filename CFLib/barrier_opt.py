from math import *
import numpy as np

try:
    from .euro_opt import cn_put, an_put, vanilla_options
except ImportError:
    from euro_opt import cn_put, an_put, vanilla_options

def cn_put_delta( S, r, T, sigma, B, M):
    mu = r - .5*sigma*sigma
    g  = 2.*mu/(sigma*sigma)
    mT = exp(-r*T)*M/S
    MT = exp(-r*T)*M*S/(B*B)
    return cn_put( T, sigma, mT) - exp( g * log(B/S)) *cn_put( T, sigma, MT)
# --

def an_put_delta( S, r, T, sigma, B, M):
    mu = r - .5*sigma*sigma;
    g  = 2.*mu/(sigma*sigma);
    mT = exp(-r*T)*M/S
    MT = exp(-r*T)*M*S/(B*B)
    return S*an_put( T, sigma, mT) - ( B*B/S)*exp( g * log(B/S)) *an_put( T, sigma, MT);
# --

def cn_put_ko( S, r, T, sigma, k, B):

    # High barrier ...
    if S < B :  
        M = min(k,B)
        return cn_put_delta( S, r, T, sigma, B,  M);

    # low barrier ...
    if k < B :  
        return 0.0;

    return cn_put_delta( S, r, T, sigma, B,  k) - cn_put_delta( S, r, T, sigma, B,  B);
# ---

def cn_call_ko( S, r, T, sigma, k, B):

    # High barrier ...
    if S < B: 
        return cn_put_delta( S, r, T, sigma, B, B) - cn_put_ko( S, r, T, sigma, k,  B);

    # low barrier ...
    mu = r - .5*sigma*sigma;
    g  = 2.*mu/(sigma*sigma);
    f  = exp( g * log(B/S)) ;
    return ( 1.0 - f )  - cn_put_delta( S, r, T, sigma, B, B) - cn_put_ko( S, r, T, sigma, k,  B);
# ---

def an_put_ko( S, r, T, sigma, k, B):

    # High barrier ...
    if S < B:
        M = min(k,B)
        return an_put_delta( S, r, T, sigma, B,  M);

    # Low barrier
    if k < B: return 0.0;
    return an_put_delta( S, r, T, sigma, B,  k) - an_put_delta( S, r, T, sigma, B,  B);
# --

def an_call_ko( S, r, T, sigma, k, B):

    # High barrier ...
    if S < B: 
        return an_put_delta( S, r, T, sigma, B, B) - an_put_ko( S, r, T, sigma, k,  B);

    # low barrier ...
    mu = r - .5*sigma*sigma;
    g  = 2.*mu/(sigma*sigma);
    f  = exp( g * log(B/S)) ;
    return ( S - (B*B/S)*f ) - an_put_delta( S, r, T, sigma, B, B) - an_put_ko( S, r, T, sigma, k,  B);
# -----

# Knock out put option
def put_ko( S, r, T, sigma, k, B):
    return exp(-r*T)*k * cn_put_ko( S, r, T, sigma, k, B) - an_put_ko( S, r, T, sigma, k, B);

# Knock out call option
def call_ko( S, r, T, sigma, k, B):
    return an_call_ko( S, r, T, sigma, k, B) - exp(-r*T)*k * cn_call_ko( S, r, T, sigma, k, B)

def run():

    S = 1.0
    r = 0.0
    T = 1.0
    sigma = .20
    k = 1.0
    Bl = .8
    Bh = 1.2

    print("%8s  %8s  %8s  %8s  %8s" %("r", "c-ko", "p-ko", "put", "call"))
    for r in np.arange(0.00, .02, .001):
        res = vanilla_options( S = S, k = k, r = r, T = T, sigma = sigma)
        c =  call_ko( S, r, T, sigma, k, Bl)
        p =  put_ko( S, r, T, sigma, k, Bh)

        print("%8.4f  %8.4f  %8.4f  %8.4f  %8.4f" %(r, c, p, res["put"], res["call"]))
# ------------------------------------

if __name__ == "__main__":
    run()
