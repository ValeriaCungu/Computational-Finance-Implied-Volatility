#!/usr/bin/env python3

from math import *
from scipy.stats import norm
import numpy as np
# -----------------------------------------------------

class HW:
    
    class_name = 'HW'

    def __init__(self, **kwargs):
        self.gamma = kwargs["gamma"] 
        self.sigma = kwargs["sigma"] 
    # --------------------


    def show(self):
        print("@ %-12s: %-8s %8.4f" %("Info", "gamma", self.gamma))
        print("@ %-12s: %-8s %8.4f" %("Info", "sigma", self.sigma))
    # --------------------

    def S2_f(self, t):
        g = self.gamma
        s = self.sigma
        h = exp( -g*t)
        # for short t
        # (t - 2t + gt^2 + t - gt^2 )/g^2
        return  ( t - (2/g)*( 1. - h ) + (1./(2*g)) * (1. - h*h ) ) /(g*g)
    # ---------------------------

    def S_X(self, t):
        g = self.gamma
        s = self.sigma
        # for short t
        # return sqrt(t)
        return sqrt( ( 1. - exp(-2*g*t))/(2*g) )
    # ---------------------------

    def cov( self, t ):
        if t < 1./(24*60):
            self.sx  = 0.0
            self.sf  = 0.0
            self.rho = 1.0
            return

        self.sx  = self.sigma*self.S_X(t)
        self.sf  = self.sigma*sqrt( self.S2_f(t) )
        self.rho = .5*( ( ( 1 - exp(-self.gamma*t) )/self.gamma )**2)/(self.S_X(t)*sqrt(self.S2_f(t)))
    # ----------------------------------------

    def BondPrice( self, t, T, dc, x):
        g = self.gamma
        b_tT = (1. - exp(-g*(T-t)))/g
        A_tT = (dc.P_0t(T)/dc.P_0t(t))*exp( .5 * (self.sigma**2)* ( self.S2_f(t) + self.S2_f(T-t) - self.S2_f(T) ) )
        return A_tT*np.exp(-b_tT*x)
    # -----------------------------------

    def Annuity( self, t, Dt, p, dc, x):
        A = np.full( len(x), 0.0, dtype=np.double ) 
        for n in range(p):
            A  += Dt*self.BondPrice( t, t+(1+n)*Dt, dc, x)
        return A
    # -----------------------------------

    def SwapRate( self, t, Dt, p, dc, x):
        A = self.Annuity(t, Dt, p, dc, x)
        R = 1. - self.BondPrice( t, t+p*Dt, dc, x)
        return R/A, A
    # -----------------------------------

    def IntPhi( self, dc, Dt, N, dt = 0.0):
        i_phi  = np.ndarray(shape = N+1, dtype=np.double)
        i_phi[0] = 0
        if dt > 0.0:
            no = 1
            #i_phi[1] = log( dc.P_0t(dt)) - .5*self.S2_f(dt) + .5 * self.S2_f(0)
            i_phi[1] = log( dc.P_0t(dt)) - .5*(self.sigma**2)* (self.S2_f(dt) - self.S2_f(0))
        else: 
            no = 0
            dt = Dt

        for n in range(no,N):
            #i_phi[n+1] = i_phi[n] + log( dc.P_0t(dt + n*Dt)/dc.P_0t( dt+(n-1)*Dt)) - .5*self.S2_f(dt+n*Dt) + .5 * self.S2_f(dt+(n-1)*Dt)
            i_phi[n+1] = i_phi[n] + log( dc.P_0t(dt + n*Dt)/dc.P_0t( dt+(n-1)*Dt)) - .5*(self.sigma**2)* ( self.S2_f(dt+n*Dt) - self.S2_f(dt+(n-1)*Dt) )

        return i_phi
# ----------------------------------------

    def Sigma( self, t, tm, T):
        '''
        Integrals in the interval [t, tm] of the
        square of the time-dependent volatility of the
        zero coupon bond P(t, T)
        '''
        g = self.gammax
        s = self.sigma
        return ( (s*s)/(2*g*g*g) ) * pow( (exp(-g*T)- exp(-g*tm)), 2) *( exp( 2*g*t) - 1 )
# ----------------------------------------

    def OptionPrice( self, t, T, Strike, dc):
        '''
        Bond put; 
        the option maturity is 't', the bond expires in T
        '''
        g = self.gamma
        s = self.sigma
        S2 = self.Sigma(t, t, T)
        
        S = sqrt( S2 )
        P_ts = dc.P_0t(t)
        P_te = dc.P_0t(T)
        F = Strike*P_ts/P_te
        if S < 1.e-08:
            if F >= 1.:
                P_an = 1.0
                P_cn = 1.0
            else:
                P_an = 0.0
                P_cn = 0.0
        else:
            Lm = log( F )/S - .5*S
            Lp = log( F )/S + .5*S
            P_an = norm.cdf(Lm)
            P_cn = norm.cdf(Lp)

        return P_ts*Strike*P_cn - P_te*P_an
# -----------------------------------
# End Class HW 
# -----------------------------------

def hw_evol(rand, hw, Dt, L, N, dt=0.0):

    '''
    Evolution of the H+W model in the 'bank-account' numeraire
    '''


    xr = rand.normal( loc = 0.0, scale = 1.0, size=(L, N))
    ir = rand.normal( loc = 0.0, scale = 1.0, size=(L, N))

    X  = np.ndarray(shape = ( L+1, N), dtype=np.double )
    Ix = np.ndarray(shape = ( L+1, N), dtype=np.double )

    X[0]  = 0.0
    Ix[0] = 0.0
    gamm = hw.gamma

    if dt > 0.0:
        lo = 1
        hw.cov(dt)
        sx   = hw.sx
        sf   = hw.sf
        rho  = hw.rho
        g    = 1 - exp(-gamm*dt)
        ir = sf*( rho*xr + sqrt( 1. - rho*rho)*ir )
        xr = sx*xr[0]

        X[1]  = xr[0]
        Ix[1] = ir[0]
    else: lo = 0


    hw.cov(Dt)
    sx   = hw.sx
    sf   = hw.sf
    rho  = hw.rho
    g    = 1 - exp(-gamm*Dt)
    ir = sf*( rho*xr + sqrt( 1. - rho*rho)*ir )
    xr = sx*xr

    for n in range(lo,L):
        mx      = - g*(X[n])
        mf      =  (g/gamm)*(X[n])
        X[n+1]  = X[n]  + mx + xr[n]
        Ix[n+1] = Ix[n] + mf + ir[n]

    return X, Ix
# ----------------------------------------

def hw_evol_P_0T(rand, hw, Dt, L, T, N, dt=0.0):

    '''
    Evolution of the H+W model in the 'terminal P_(t,T)' numeraire
    This evolution is legal for t <= L Dt.

    We only handle trajectories where all steps are Dt with possibly
    the first one to be of a different length.

    rand: rand object
    hw  : HW-object
    DT  : trajectory step
    L   : number of steps in the trajectory
    T   : the numeraire is P(t, T)
    N   : number of trajectories
    dt  : a possible first step different from other steps
    '''


    xr = rand.normal( loc = 0.0, scale = 1.0, size=(L, N))
    X  = np.ndarray(shape = ( L+1, N), dtype=np.double )
    X[0]  = 0.0

    #
    # a possible first step different from all of the others
    #
    if dt > 0.0: 
        l = 1
        #sx   = sqrt(hw.S2_X(dt))
        gamm = hw.gamma
        sgma = hw.sigma
        sx   = sgma*hw.S_X(dt)
        g    = exp(-gamm*dt)
        h    = (1. - exp(-gamm*dt))/gamm
        h2   = (1. - exp(-2*gamm*dt))/(2*gamm)
        mx      = - ((sgma*sgma)/gamm)*(h - exp(-gamm*(T-dt))*h2)
        X[1]  = mx + sx*xr[0]
    else: l = 0

    #sx   = sqrt(hw.S2_X(Dt))
    gamm = hw.gamma
    sgma = hw.sigma
    sx   = sgma*hw.S_X(Dt)
    g    = exp(-gamm*Dt)
    h    = (1. - exp(-gamm*Dt))/gamm
    h2   = (1. - exp(-2*gamm*Dt))/(2*gamm)
    xr   = sx*xr


    for n in range(l, L):
        mx      = g*X[n] - ((sgma*sgma)/gamm)*(h - exp(-gamm*(T-(n+1)*Dt))*h2)
        X[n+1]  = mx + xr[n]

    return X
# ----------------------------------------
