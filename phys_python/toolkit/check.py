# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 16:07:23 2021

@author: Yuhan Liu
"""
import numpy as np
import matplotlib.pyplot as plt



def check_diag(matr, is_show = 1):
    matr_remove = matr-np.diag(np.diag(matr))
    diag_error = np.sum(abs(matr_remove))
    
    if (diag_error > 10**(-6) and is_show == 1):
        plt.imshow(abs(matr), cmap = 'jet')
        plt.colorbar()
        # plt.rcParams["figure.figsize"] = (10,10)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        plt.show()
          
    return diag_error


def check_identity(matr, is_show = 1):
    
    matr_remove = matr-np.eye(len(matr))
    id_error = np.sum(abs(matr_remove))
    
    if (id_error > 10**(-6) and is_show == 1):
        plt.imshow(abs(matr), cmap = 'jet')
        plt.colorbar()
        plt.show()
           
    return id_error
    

def check_zero(matr, is_show = 0):
    
    zero_error = np.sum(abs(matr))
    
    if (zero_error > 10**(-6) and is_show == 1):
        plt.imshow(abs(matr), cmap = 'jet')
        plt.colorbar()
        plt.show()
           
    return zero_error


def check_hermitian(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.conj().T, rtol=rtol, atol=atol)


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def check_real(matr):    
    real_error = np.sum(abs(matr.imag))
    return real_error