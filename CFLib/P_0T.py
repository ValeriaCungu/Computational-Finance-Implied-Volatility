#!/usr/bin/env python3

import sys
from sys import stdout as cout
from scipy.stats import norm
from math import *
import numpy as np
import pandas as pd
from time import time
try:
    from CFLib.stats import stats
    from CFLib.config import get_input_parms, loadConfig
    from CFLib.finder import find_pos
except ModuleNotFoundError:
    from stats import stats
    from config import get_input_parms, loadConfig
    from finder import find_pos
# -----------------------------------------------------

class Zc:

    def __init__( self, **keywrds):
        curve   = keywrds["curve"]
        self.tl = curve[0]
        self.rc = curve[1]
        self.pt = np.exp(-self.tl*self.rc)
    # ------------------------------------------

    @classmethod
    def from_discount_curve(cls, t, P):
        if t[0] == 0:
            t = np.array(t[1:])
            P = P[1:]
        r = -np.log(P)/t
        return cls(curve=(t, r))

    @classmethod
    def from_cc_zero_coupon_rates(cls, t, rc):
        return cls(curve=(t, rc))

    @classmethod
    def from_yc_zero_coupon_rates(cls, t, yc):
        r = np.log( 1 + yc)
        return cls(curve=(t, r))


    def f_0t( self, t ):

        '''
            f_0t := \int_0^t r(s) ds
            
            condition: t >= 0
        '''

        tl  = self.tl
        r  = self.rc
        pt = self.pt

        n = find_pos( t, tl )

        if n < 0            : return t*r[0]
        if n == tl.size - 1 : return t*r[-1]
        
        fs = tl[n]*r[n]
        fe = tl[n+1]*r[n+1]
        return (tl[n+1] - t) * fs/(tl[n+1] - tl[n]) + (t - tl[n])* fe/(tl[n+1]-tl[n])
    #------------------------------------------------

    def rz( self, t): return self.f_0t(t)/t
    def ry( self, t): return exp(self.f_0t(t)/t) - 1.
    def P_0t( self, t): return exp( -self.f_0t(t) )
    # -----------------------------------------------------------------

    
    def swap_rate( self, tm=0.0, p=10, Dt=1.0):
        '''
        returns swap rate R_p(tm)
        and annuity       A_p(tm)

        where
        R_p(tm) = [ P(0, t_m) - P(0, t_m + p  Dt) ]/A_p(t_m)
        A_p(tm) = Sum_[1 <= j <= p] Dt P(0,t_m+j*Dt )
        '''
        A  = 0.0

        for n in range(1,p+1): A += Dt*self.P_0t( tm + n*Dt)
        Num = self.P_0t(tm) - self.P_0t( tm+p*Dt)
        return Num/A, A


    def show( self ):

        tl = self.tl
        n  = 0
        print("%3s  %9s  %8s  %8s  %8s" %( "pos", "t", "P_0t", "rc", "ry"))
        for t in tl:
            if fabs(t) < 1.e-10: continue
            p  = self.P_0t(t)
            xc = self.rz(t)
            xy = self.ry(t)
            print("%3d  %9.6f  %8.6f  %8.6f  %8.6f" %( n, t, p, xc, xy))
            n += 1
        


# -----------------------------------------------------

def usage():
    print("Usage: $> python3 euro_opt.py [options]")
    print("Options:")
    print("    %-24s: this output" %("--help"))
    print("    %-24s: output file" %("-out"))
    print("    %-24s: input file" %("-in"))
# -----------------------------------------------------


def run(argv):

    output = None
    parms  = get_input_parms(argv)
    print("@ %-12s: %s" %("argv", str(argv)));
    print("@ %-12s: %s" %("parms", str(parms)));

    try:
        Op = parms["help"]
        usage()
        return
    except KeyError:
        pass

    try:
        output = parms["out"]
        fp = open(output, "w")
    except KeyError:
        fp = cout

    inpt = parms["in"]
    PAR    = loadConfig(inpt)

    dc = Zc(curve = PAR.curve)
    # dc.show()

    D = 1./365.
    W = 1./52.
    M = 1./12.
    Y = 1.
    times = [ 1*D, 2.*D, 3*D, 4.*D, 5.*D, 6*D, 1.*W, 2.*W, 1.*M, 2.*M, 3.*M, 6.*M, 1.*Y, 2.*Y, 5.*Y, 10.*Y, 15.*Y, 20.*Y, 30.*Y]

    n = 0
    fp.write("\n")
    fp.write("%3s  %9s  %8s  %8s  %8s\n" %("n", "t", "P_0t", "r", "ry") )
    for t in times:
        fp.write("%3d  %9.6f  %8.6f  %8.6f  %8.6f\n" %(n, t, dc.P_0t(t), dc.rz(t), dc.ry(t)) )
        n += 1

    print("#-\n")
    dc.show()

   
    if output != None:
        print("@ %-12s: output written to '%s'\n" %("Info", output))
        fp.close()
# --------------------------

if __name__ == "__main__":
    run(sys.argv)
