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
    
    def __init__(self,ent_type, NAB,  N = 0, renyi = 0, var2 = 0):
        
        """
        var2 is added, in case that we need to loop over another variable 
        other than the system size.
        """
        
        self.renyi = renyi
        self.ent_type = ent_type
        self.NAB = NAB
        
        if N != 0:
            self.N = N
        if not isinstance(var2,int):
            self.var2 = var2
        
        if isinstance(NAB,int):
            pts = 1
        elif isinstance(var2,int):
            pts = len(NAB)
        else:
            pts = (len(NAB),len(var2))
        
        self.SAB = np.zeros(pts,dtype=np.complex128)
        
        if renyi>0:
            self.RenyiAB = np.zeros(pts,dtype=np.complex128)
        
        if (2 in ent_type):
            self.LN = np.zeros(pts,dtype=np.complex128)
            
        if (3 in ent_type):
            self.RE = np.zeros(pts,dtype=np.complex128)
            if renyi>0:
                self.RE_ry = np.zeros(pts,dtype=np.complex128)
            
        if (4 in ent_type):
            self.MI = np.zeros(pts,dtype=np.complex128)
            self.SA = np.zeros(pts,dtype=np.complex128)
            self.SB = np.zeros(pts,dtype=np.complex128)
            
            if renyi>0:
                self.MI_ry = np.zeros(pts,dtype=np.complex128)
      
    
    def update_ent(self, ent, pt1, pt2 = 0):
        
        renyi = self.renyi
        ent_type = self.ent_type
        
        if pt2 == 0:
            pt = pt1
        else:
            pt = (pt1,pt2)
        
        self.SAB[pt] += ent.S
        
        if renyi>0:
            self.RenyiAB[pt] += ent.Renyi
            
        if (2 in ent_type):
            self.LN[pt] += ent.LN
            
        if (3 in ent_type):
            self.RE[pt] += ent.RE
            if renyi>0:
                self.RE_ry[pt] += ent.RE_ry
            
        if (4 in ent_type):
            self.MI[pt] += ent.MI
            self.SA[pt] += ent.SA
            self.SB[pt] += ent.SB
            
            if renyi>0:
                self.MI_ry[pt] += ent.MI_ry
                
                
    def __add__(self, other):
        
        
        if isinstance(other, int):
            # This is to deal with the case where other = 0. Return itself
            return self
        
        if len(self.NAB) != len(other.NAB):
            print(" --! Check your input")
            return 0
        
        ent_type = self.ent_type
        
        self.SAB += other.SAB
        
        if self.renyi>0:
            self.RenyiAB += other.RenyiAB
            
        if (2 in ent_type):
            self.LN += other.LN
            
        if (3 in ent_type):
            self.RE += other.RE
            if self.renyi>0:
                self.RE_ry += other.RE_ry
            
        if (4 in ent_type):
            self.MI += other.MI
            self.SA += other.SA
            self.SB += other.SB
            
            if self.renyi>0:
                self.MI_ry += other.MI_ry
                
        return self
    
                
    def avg(self, sample_n):
        
        ent_type = self.ent_type
        
        self.SAB = self.SAB/sample_n
        
        if self.renyi>0:
            self.RenyiAB = self.RenyiAB/sample_n
            
        if (2 in ent_type):
            self.LN = self.LN/sample_n
            
        if (3 in ent_type):
            self.RE = self.RE/sample_n
            if self.renyi>0:
                self.RE_ry = self.RE_ry/sample_n
            
        if (4 in ent_type):
            self.MI = self.MI/sample_n
            if self.renyi>0:
                self.MI_ry = self.MI_ry/sample_n
        
        