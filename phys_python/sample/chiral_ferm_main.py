# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 16:52:44 2022

@author: Yuhan Liu (yuhanliu@uchicago.edu)
"""

import numpy as np
from math import pi

from hamiltonian.ferm_tool import Ferm_hamiltonian
from entangle.ent_generate import GenerateEnt
from entangle.ent_ferm import GetEntFerm
from toolkit.plot_style import plot_s


CUT_OFF = 10**(-8)

def check_es(Corr_r):

    eig_c,_ = np.linalg.eig(Corr_r)
    eig_c = np.sort(eig_c.real)
    
    eig_c = eig_c[eig_c>CUT_OFF]
    eig_c = eig_c[eig_c<1-CUT_OFF]
    
    es = np.log((1-eig_c)/eig_c)
    
    return es


def get_ent(Corr, Gamma, NAB, d, N, bands, renyi = 0, 
                is_non_herm = 0, PT = 'true'):
    
    # ent_type = [1,2,3,4]
    ent_type = [1,2,3,4]
      
    result_ent = GenerateEnt(ent_type, NAB, N, renyi = renyi)
    
    for i in range(len(NAB)):
        LAB = NAB[i]
        
        total_sub = int(2*LAB*(1+d))
        if total_sub < 2*N:
            GammaR = Gamma[0: total_sub,0: total_sub]
    
            ent = GetEntFerm(GammaR = GammaR, ent_type = ent_type, 
                            renyi = renyi,
                            partition = [2*int(LAB/2),2*int(LAB/2), int(2*d*LAB)], 
                            is_non_herm = is_non_herm, PT = PT)
            
            result_ent.update_ent(ent,i)
        else:
            print("out of range")
        
    return result_ent



def main_fun(N = 14, d = 0):
    
    if d == 0:
        N_sub = range(2, N-2, 2)
        # N_sub = range(40,60,2)
    elif d==0.5:
        N_sub  = range(2, int(N/3*2)-1)

    
    H = np.zeros((N,N))
    PBC = -1
    
    for i in range(N-1):
        H[i,i+1] = 1
        H[i+1,i] = 1
        
    H[0, N-1] = PBC*1
    H[N-1, 0] = PBC*1
    
    chiral = Ferm_hamiltonian(N = N, H = H, P = PBC)
    eigval, eigvec = chiral.single_eig()
    
    # eig_benchmark = np.cos(2*pi/N*np.arange(0,N)+(PBC-1)/2*pi/N)*2
    
    m_fill = np.hstack((np.ones(int(N/2)), np.zeros(int(N/2))))
    
    # # Change the fermi surface. If I use N = 80, it still gives c=1.0195
    # m_fill = np.hstack((np.ones(int(N/4)), np.zeros(int(3*N/4))))
    
    Corr, Gamma = chiral.get_gamma(m_fill)
    res_ent = get_ent(Corr,Gamma,N_sub, d, N,1)
    
    
    if d == 0:
        # plot_s(N_sub, res_ent.LN.real, N = N, fit_type=3)
        plot_s(N_sub, res_ent.SAB.real, N = N, fit_type=1)
        
        
    elif d == 0.5:
        usr_func = lambda x, c, c1: c/3*(3*np.log(np.sin(pi*x/2/N))
                                         -2*np.log(np.sin(2*pi*x/2/N))
                                         +np.log(np.sin(3*pi*x/2/N)))+c1
        plot_s(N_sub, res_ent.SAB.real, N = N, fit_type=-1, usr_func=usr_func)
    else:
        plot_s(N_sub, res_ent.LN.real, N = N, fit_type=3)

    return N_sub, res_ent
    
    
if __name__ == "__main__":
    # d = 0 or 0.5
    d = 0
    N_sub, res = main_fun(300, d= d)
    
    