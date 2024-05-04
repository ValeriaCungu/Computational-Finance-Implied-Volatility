import math
import numpy as np
from math import *

def pr_x_lt_w ( model, Xc, vW, off, t):

    '''
        Digital option according to the SINC algorithm
    '''

    m = 1
    vTot = 0.0
    while True:
        c_k    = 2*math.pi*( m/(2*Xc) + off )
        c_phi  = model.cf(c_k, t)
        vTh    = math.pi * m * vW/Xc;
        vDelta = (np.cos(vTh)*c_phi.imag - np.sin(vTh)*c_phi.real)/m; 
        vTot  += vDelta
        if np.fabs(vDelta/vTot).max() < 1.e-08: break
        m += 2
    return .5 - 2.*vTot/math.pi
# --------------------------------------------------------------------

'''
Computes the price of vanilla options for the model 'model'

model   : model object that admits the method 'cf', returning the complex array
          describing the characteristic function of the model
vSTrike : an np.array of strikes for the options
T       : the maturity of the options involved
Xc      : The Xc associated to this maturity
'''
def ft_opt(model, vStrike, T, Xc):

    vW       = np.log(vStrike)

    #
    # cash or nothing option in the terminla measure
    #
    off = complex(0.0, 0.0)
    vCn = pr_x_lt_w( model, Xc, vW, off, T)

    #
    # cash or nothing option in the S(T) measure
    #
    off = complex(0.0, -1/(2*math.pi))
    vAn = pr_x_lt_w( model, Xc, vW, off, T)

    
    vPut  = vStrike*vCn - vAn; 
    vCall = vPut + (1. - vStrike);

    return {"put": vPut, "call":  vCall, "pCn": vCn, "pAn": vAn}
# --------------------------------------------------------------------
