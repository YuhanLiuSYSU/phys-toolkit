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


def fit_ent(interval, ent, N = 0, renyi = 1, fit_type = 1, usr_func = 0, p0 = None):
    """
    I recommend to use this in plot_s.py
    
    Example usage:
        x_data = [12,14,16,18,20]
        y_data = [0.7424, 0.7344, 0.7275, 0.7215, 0.7162]
        
        usr_func = lambda x, a, b, c: b*x**(-a)+c
        coeffs = plot_style_s(x_data, y_data,
                              x_labels = "$L$", y_labels = "$III$",
                              fit_type = -1, usr_func = usr_func)
    """
    
    Fit_func = FitClass()
    Fit_func.renyi = renyi
    Fit_func.N = N
    
    if isinstance(interval,range):
        interval = np.array(interval)
    
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
    
    coeffs, coeffs_cov = curve_fit(my_func,interval,ent, p0 = p0)
    
    return coeffs, coeffs_cov, my_func