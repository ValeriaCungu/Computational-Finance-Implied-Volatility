
import cmath
from math import *
import numpy as np
from CFLib.FT_opt import ft_opt


class VG:

    def __init__(self, **kwargs):
        self.eta    = kwargs["eta"]
        self.nu     = kwargs["nu"]
        self.th     = kwargs["theta"]

        self.Phi()
    #-----------------

    def Phi(self):
        self.phi = log( 1. - self.nu*self.th - .5*self.nu*self.eta*self.eta)

    def get(self):
        return np.array([self.eta, self.nu, self.th]) 

    def set(self, x):
        self.eta   = x[0]
        self.nu    = x[1]
        self.th    = x[2]
        self.Phi()

    @property
    def intensity(self): return 1./self.nu
    def compensator(self): return self.phi

    def cf(self, c_k, t):
        # 
        # c_u = i c_k
        #
        c_u = c_k*1j

        c_x = cmath.log( 1.0 -self.nu*self.th*c_u -.5*self.nu*(self.eta*self.eta)*c_u*c_u)
        comp = self.compensator()
        JMP  = t*self.intensity*(comp*c_u - c_x)

        return cmath.exp(JMP)

    def VGPut( self
             , So     = 1.0
             , Strike = 1.0
             , T      = 1.0
             , Xc     = 1.
             , r      = 0.0
             , q      = 0.0
             ):
    
        Fw = So*exp((r-q)*T)
        return exp(-r*T)*Fw*ft_opt(self, Strike/Fw, T, Xc*sqrt(T))["put"]

# -----------------------------------------------------

def vg_evol_step( rand, Sn, vg, Dt, N ):
    nu  = vg.nu
    eta = vg.eta
    th  = vg.th
    I   = vg.intensity
    phi = vg.compensator()
    g   = rand.normal( loc = 0.0, scale = 1.0, size=(N))
    xi  = np.float64( rand.gamma(shape=Dt/nu, scale=nu, size=(N) ))
    X   = th*xi + eta*g*np.sqrt(xi) + Dt*I*phi
    return Sn*np.exp(X)

def vg_evol( rand, So, vg, Nt, Dt, N ):

    S  = np.ndarray(shape = (Nt+1, N), dtype=np.double ) # S[N, L] in fortran matrix notation
    S[0] = So
    for n in range(Nt):
        S[n+1] = vg_evol_step(rand, S[n], vg, Dt, N)

    return S

# ----------------------------------------------------
