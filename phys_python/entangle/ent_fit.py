# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 16:09:59 2021

@author: Yuhan Liu
"""
import numpy as np
from math import pi
from scipy.optimize import curve_fit



class FitClass:
    " Various fit functions for entanglement. "
    
    def __init__(self):
        pass
    
    def fit_func_ent(self,x,c,c1):
        # for entanglement entropy
        
        coe = (self.renyi+1)/(6*self.renyi)
        #coe = 1/3 ### debug
        return c*coe*np.log(np.sin(pi*x/self.N))+c1
    
    def fit_func_MI(self,x,c,c1):
        # for mutual information
        
        return c/3*np.log((np.sin(pi*x/(2*self.N)))**2/np.sin(pi*x/self.N))+c1
    
    def fit_func_LN(self,x,c,c1):
        # for logarithmic negativity
        return c/4*np.log((np.sin(pi*x/(2*self.N)))**2/np.sin(pi*x/self.N))+c1


def fit_ent(interval, ent, N = 0, renyi = 1, fit_type = 1, usr_func = 0):
    Fit_func = FitClass()
    Fit_func.renyi = renyi
    Fit_func.N = N
    
    if fit_type == 0:
        my_func = lambda x, a, b : a*x+b
    elif fit_type == 1:
        my_func = Fit_func.fit_func_ent
    elif fit_type == 2:
        my_func = Fit_func.fit_func_MI    
    elif fit_type == 3:
        my_func = Fit_func.fit_func_LN
    else:
        my_func = usr_func
    
    
    coeffs, coeffs_cov = curve_fit(my_func,interval,ent)
    
    return coeffs, coeffs_cov, my_func