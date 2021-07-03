import numpy as np
import h5py

import math

def func(rho):
    return (8 / 3) *  math.pi * rho * ((1/2.5)** 9 * (1/3) -(1/2.5)**3)

i=0
while i< 10:
    i+=1
    print(i)
    if i == 6:
        break
