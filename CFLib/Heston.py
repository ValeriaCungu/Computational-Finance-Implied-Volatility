import math
import cmath
from math import *
import numpy as np
from CFLib.FT_opt import ft_opt

#      1 + g
#  ---------------------
#    1 + g exp( - gamma * T )
#
def arg_log( c_gmma, c_g, T):
    return (1.0 + c_g)/(1.0 + c_g*np.exp(-c_gmma*T))

#
#  C = 2 * log( (1 - g) / ( 1 - g e^{-gamma T } ) )/sigma^2
#  arg_log := (1 - g) / ( 1 - g e^{-gamma T } )
#
def C( c_gmma, c_g, sigma, T):
    c_psi = arg_log(c_gmma, c_g, T)
    return 2*np.log(c_psi)/pow(sigma,2)

#
# .5 ( v - v^2 )
#
def Lambda( c_v): 
    return .5*c_v - .5*c_v*c_v

#
# Gamma =  SQRT( kappa^2 + 2 L sigma^2 )
#
def Gamma( c_L, c_kappa, sigma):
    return cmath.sqrt(c_kappa*c_kappa + 2.*c_L*pow(sigma,2))
#
# g = ( gamma - kappa )/( gamma + kappa );
#
def G( c_kappa, c_gmma):
    return (c_gmma - c_kappa)/(c_gmma + c_kappa)

#
# Z_p = ( gamma - kappa )/sigma^2
#
def Z_p( c_gmma, c_kappa, sigma):
    return (c_gmma - c_kappa)/pow(sigma,2);
#
#
# A = -kappa * theta * ( zp * T - C)
#
def A_tT( c_gmma, c_g, c_zp, c_kappa, c_theta, sigma, T):
    c_c  = C( c_gmma, c_g, sigma, T)
    c_kt = c_kappa * c_theta
    return c_kt*(c_c - c_zp*T)
#
#
# B = zp * ( 1 - e^{-gamma T } )/( 1 + g e^{-gamma T} )
#
def B_tT( c_gmma, c_g, c_zp, T):
    c_exp_gt = np.exp(-c_gmma*T)
    return c_zp*(1. - c_exp_gt)/( 1. + c_g*c_exp_gt )



class Heston:

    def __init__(self, **kwargs):
        self._lambda= kwargs["lmbda"] 
        self._nubar = kwargs["nubar"] 
        self._eta   = kwargs["eta"] 
        self._nu_o  = kwargs["nu_o"] 
        self._rho   = kwargs["rho"]

    #def __init__(self, x):
    #    self._lmbda = x[0]
    #    self._nubar = x[1]
    #    self._eta   = x[2]
    #    self._nu_o  = x[3]
    #    self._rho   = x[4]
# ----------------------------------------

    @property
    def lmbda(self): return self._lambda
    @property
    def nubar(self): return self._nubar
    @property
    def eta(self)  : return self._eta
    @property
    def nu_o(self) : return self._nu_o
    @property
    def rho(self) : return self._rho

    def get(self):
        return np.array([self._lambda, self._nubar, self._eta, self._nu_o, self._rho]) 

    def set(self, x):
        self._lambda = x[0]
        self._nubar = x[1]
        self._eta   = x[2]
        self._nu_o  = x[3]
        self._rho   = x[4]

    def log_cf(self, c_k, t): 
        if c_k.real == 0.0: return 0. + 0j

        c_v     = c_k*1j
        c_L     = Lambda(c_v)
        c_kappa = self.lmbda - self.rho*self.eta*c_v
        c_gmma  = Gamma( c_L, c_kappa, self.eta)
        c_g     = G    ( c_kappa, c_gmma)
        c_zp    = Z_p  ( c_gmma, c_kappa, self.eta)

        if self.lmbda == 0.0:
            c_A = 0. + 0.0j
        else:
            c_theta = (self.lmbda*self.nubar)/c_kappa;
            c_A = A_tT( c_gmma, c_g, c_zp, c_kappa, c_theta, self.eta, t)

        c_B = B_tT ( c_gmma, c_g, c_zp, t)

        return c_A - self.nu_o*c_B
    # -------------------------------------

    def cf(self, c_k, t):
        c_x = self.log_cf( c_k, t)
        return np.exp(c_x)
    # -----------------------

    def HestonPut( self
                 , So     = 1.0
                 , Strike = 1.0
                 , T      = 1.0
                 , Xc     = 1.
                 , r      = 0.0
                 , q      = 0.0
                 ):
    
        Fw = So*exp((r-q)*T)
        return exp(-r*T)*Fw*ft_opt(self, Strike/Fw, T, Xc*sqrt(T))["put"]
    

    def HestonCall( self
                 , So     = 1.0
                 , Strike = 1.0
                 , T      = 1.0
                 , Xc     = 1.
                 , r      = 0.0
                 , q      = 0.0
                 ):
    
        Fw = So*exp((r-q)*T)
        return exp(-r*T)*Fw*ft_opt(self, Strike/Fw, T, Xc*sqrt(T))["call"]
