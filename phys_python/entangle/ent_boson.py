# -*- coding: utf-8 -*-
"""
Bosonic entanglement quantities, using covariance method

Created on Mon Nov  1 16:56:40 2021
@author: Yuhan Liu
"""

import numpy as np
import scipy.linalg as alg

from toolkit.check import check_real


# TODO: benchmark using harmonic oscillator

class GetEntBoson:
    
    def __init__(self, sigma = 0, X = 0, P = 0, partition = 0):
                        
        if (partition != 0):
            NA = partition[0]
            NB = partition[1]
            self.NA = NA
            self.NB = NB
            if (len(partition) > 2):
                d = partition[3]
                self.d = d
            else:
                d = 0
                self.d = d    
              
        if isinstance(sigma, np.ndarray):
            self.sigma = sigma
            # TODO: property of sigma? Postive eigenvalue? 
                    
            # N is the size of the system (number of dof)
            N = int(len(sigma)/2)
            omega = np.kron(np.array([[0,1],[-1,0]]), np.eye(N))
            J = -1j*sigma@omega
    
            mu = alg.eig(J)[0]
                       
        if isinstance(X, np.ndarray):            
            C = X @ P
            mu = np.sqrt(alg.eig(C)[0])
            
            
        is_real = check_real(mu)
        if is_real>10**(-6): 
            print(" --! [GetEntBoson] mu real error: ", str(is_real))
        
        self.mu = mu.real
            
        self.S = self.get_S_()
        
        
        
    def get_S_(self):
        mu = self.mu.real
        mu = mu[mu > 1+10**(-8)]
        return ((mu/2+1/2)*np.log(mu/2+1/2) - (mu/2-1/2)*np.log(mu/2-1/2)).sum()
        
        