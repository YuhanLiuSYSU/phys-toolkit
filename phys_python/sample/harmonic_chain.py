# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 19:32:44 2021

@author: Yuhan Liu (yuhanliu@uchicago.edu)
"""

import numpy as np
from math import pi, sin

from entangle.ent_generate import GenerateEnt
from entangle.ent_boson import GetEntBoson
from toolkit.plot_style import plot_style_s


def harmonic_chain(L = 100):

    omega = 10**(-5)
    ent_type = [1]
    
    LA = np.arange(1, 40+1)
    
    posi_array = np.array([[i-j for i in range(1,L+1)] 
                                for j in range(1,L+1)])
    omegak = lambda k: np.sqrt(omega**2+4*(sin(pi*k/L))**2)
    
    X = np.zeros((L,L))
    P = np.zeros((L,L))
    zero_block = np.zeros((L,L))
    
    for k in range(1, L):
        cos_fac = np.cos(2*pi*k/L*posi_array)
        X = X + 1/(2*L*omegak(k))*cos_fac
        P = P + omegak(k)/(2*L)*cos_fac
    
    X = X + 1/(2*L*omegak(0))
    P = P + omegak(0)/(2*L)
    
    result_ent = GenerateEnt(ent_type, 2*LA, L)
    
    for i in range(len(LA)):
        lA = LA[i]
        lAB = 2*lA
        
        # sigmaR = np.hstack((np.vstack((2*X[0:lAB, 0:lAB], zero_block[0:lAB, 0:lAB])), 
        #                     np.vstack((zero_block[0:lAB, 0:lAB],2*P[0:lAB, 0:lAB]))))
        
        # ent = GetEntBoson(sigma = sigmaR, partition = [lA, lA])
        ent = GetEntBoson(X = 2*X[0:lAB, 0:lAB], P = 2*P[0:lAB, 0:lAB], partition = [lA,lA])
        result_ent.update_ent(ent,i)
        
    return result_ent
    

if __name__ == "__main__":

    result = harmonic_chain()
    x_data = result.NAB
    y_data = result.SAB.real
    
    plot_style_s(x_data,y_data, N = result.N, fit_type = 1)