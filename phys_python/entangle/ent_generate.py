# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 16:38:39 2021

@author: Yuhan Liu
"""
import numpy as np



class GenerateEnt:
    """ 
    Generate the entanglement data object   
    """
    
    def __init__(self,ent_type, NAB, N = 0, renyi = 0):
        
        self.renyi = renyi
        self.ent_type = ent_type
        self.NAB = NAB
        
        if N != 0:
            self.N = N
        
        if isinstance(NAB,int):
            pts = 1
        else:
            pts = len(NAB)
        
        self.SAB = np.zeros(pts,dtype=np.complex128)
        
        if renyi>0:
            self.RenyiAB = np.zeros(pts,dtype=np.complex128)
        
        if (2 in ent_type):
            self.LN = np.zeros(pts,dtype=np.complex128)
            
        if (3 in ent_type):
            self.RE = np.zeros(pts,dtype=np.complex128)
            
        if (4 in ent_type):
            self.MI = np.zeros(pts,dtype=np.complex128)
      
    
    def update_ent(self, ent, pt):
        
        renyi = self.renyi
        ent_type = self.ent_type
        
        self.SAB[pt] = ent.S
        
        if renyi>0:
            self.RenyiAB[pt] = ent.Renyi
            
        if (2 in ent_type):
            self.LN[pt] = ent.LN
            
        if (3 in ent_type):
            self.RE[pt] = ent.RE
            
        if (4 in ent_type):
            self.MI[pt] = ent.MI