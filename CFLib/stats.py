from math import *
import numpy as np 

def stats(x):

    '''
    Given the constant array 'x', stats will return the tuple
    ( E[x], StDev(x) := sqrt( E[ (x - E[x])^2 ] ))
    where E, represents the sample average.
    '''

    return x.mean(), x.std()
