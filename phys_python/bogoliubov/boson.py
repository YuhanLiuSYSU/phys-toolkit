# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:04:34 2021

@author: Yuhan Liu
"""
# import sys
# sys.path.append('C:\\Users\\sglil\\OneDrive\\Desktop\\phys-toolkit\\phys-python')

import scipy.linalg as alg
from numpy.linalg import inv
import numpy as np
from phys_python.common import check_zero, sort_ortho


class BosonBogoliubov:
    
    def __init__(self, h):
        self.h = h
        
        N = len(h)
        sz = np.array([[1,0],[0,-1]])
        g = np.kron(sz, np.eye(int(N/2)))     
        self.g = g
        
        
    def decomp(self):
        
        h = self.h
        g = self.g
        N = len(h)
        
        
        e_val, e_vec = sort_ortho(h)        
        e_vec = e_vec.conj().T
        
        if any(e_val < 0):
            print(min(e_val))
            print(" --! Error: not positive definite")
            return 0, 0 
  
        k = np.diag(np.sqrt(e_val)) @ e_vec
        hp = k @ g @ k.conj().T
        
        is_decompk = check_zero(k.conj().T@k - h)
        if (is_decompk > 10**(-8)):
            print(" --! Error: k decomposition fails: "+str(is_decompk))
            return 0, 0
            
        L, U = sort_ortho(hp)
        idx = L.argsort()[::-1]   
        L = L[idx]
        U = U[:,idx]
        
        if N>2:
            # sort the second half
            Lm = L[int(N/2):N]
            Um = U[:,int(N/2):N]
            
            idx = Lm.argsort()[::1] 
            Lm = Lm[idx]
            Um = Um[:,idx]
    
            L[int(N/2):N] = Lm
            U[:,int(N/2):N] = Um        
        
        E = g @ np.diag(L)
        T = inv(k) @ U @ np.sqrt(E)
        
        Ta = T[0:int(N/2),0:int(N/2)]
        Tb = T[int(N/2):N,0:int(N/2)]
        
        T[0:int(N/2),int(N/2):N] = Tb.conj()
        T[int(N/2):N,int(N/2):N] = Ta.conj()
        
        S = (Tb.conj()) @ inv(Ta.conj())
        
        self.E = E
        self.T = T
        self.S = S
 
        return self.E, self.T
    
    def sanity_check(self):
        
        T = self.T
        E = self.E
        g = self.g
        h = self.h
        
        print(" --> Error for decomposition: " 
              + str(check_zero(T.conj().T @ h @ T - E)))
        print(" --> Error for commutator: " 
              + str(check_zero(T.conj().T @ g @ T - g)))
        
        
        numerics_re = np.sort(np.diag(E))
        expected_re = np.sort(abs(alg.eig(g @ h)[0]))
        if len(h) < 20:
            print(" --> numerics: ")
            # print(np.diag(E))
            print(numerics_re)
            print(" --> expected: ")
            # g*h is not hermitian, so we need to use alg.eig
            print(expected_re)
        else:
            print(" --> numerics - expected", str(check_zero(numerics_re-expected_re)))
        

if __name__ == "__main__":
    
    # Benchmark
    model = 2
    
    if model == 1:
        ep = 1
        ld = 0.5
        
        h = np.array([[ep,ld],[ld,ep]])
        
    else:    
        ll = 5
        cm = 1-ll
        cp = 1+ll
        ep = 0.0000001
        
        h = np.array([[cp+ep,-cp,cm,-cm],
                      [-cp,cp+ep,-cm,cm],
                      [cm,-cm,cp+ep,-cp],
                      [-cm,cm,-cp,cp+ep]])
    
        
    hbdg = BosonBogoliubov(h)
    [E,T] = hbdg.decomp()
    hbdg.sanity_check()
    
    overlap = T[2,0]**2+T[2,1]**2
    print()
    print("expected value is: " + str(overlap))
    

    
        