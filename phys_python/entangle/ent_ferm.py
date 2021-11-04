"""
Created on Fri Oct 22 13:15:56 2021

@author: Yuhan Liu (yuhanliu@uchicago.edu)
"""
import numpy as np
import scipy.linalg as alg
import cmath

from eig.decomp import sort_biortho, sort_ortho, decomp_schur_


# TODO: check tr(Gamma_c)=0
# Q: check eig or eigh? 
# A: Doesn't matter here because we only need eigval.


class GetEntFerm:
    """ 
    Class for the entanglement data for single body problem.
    The input is either covariance matrix or correlation matrix.  
    
    """
    
    def __init__(self, GammaR = 0, corr = 0, ent_type = 1, partition = 0, is_non_herm = 0, renyi = 0, PT = 'true'):
        self.GammaR = GammaR
        self.corr = corr
        self.ent_type = ent_type
        self.is_non_herm = is_non_herm
        self.PT = PT  
        self.renyi = renyi
        
        self.eig_gamma, _ = alg.eig(GammaR)
               
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
             
        if any(y > 1 for y in ent_type):
            G11 = GammaR[0:NA, 0:NA]
            G12 = GammaR[0:NA, d+NA:d+NA+NB]
            G21 = GammaR[d+NA:d+NA+NB, 0:NA]
            G22 = GammaR[d+NA:d+NA+NA, d+NA:d+NA+NB]
            
            if d > 0:
                self.GammaR = np.vstack((np.hstack((G11,G12)),
                                         np.hstack((G21,G22))))
            
            
        if isinstance(corr,int):
            self.S = self.get_S_(self.eig_gamma)
        else:
            self.S = self.get_SCorr_(self.corr)
            
        if self.renyi > 0:
            fac = 1/(1-self.renyi)
            self.Renyi = fac*self.get_renyi_(self.renyi, self.eig_gamma)   
                   
        
        if (2 in ent_type):
            self.LN = self.get_LN_(G11,G12,G21,G22);
        
        if (3 in ent_type):
            # when there is corr input...
            if (type(self.corr) != int):
                self.RE = self.get_RECorr_();    
            else:
                self.RE = self.get_RE_();
                
            
        if (4 in ent_type):
            
            eig_G11, _ = alg.eig(G11)
            eig_G22, _ = alg.eig(G22)
            
            self.MI = self.get_MI_(eig_G11, eig_G22)

            
        
    " von Neumann entropy "
    def get_S_(self,gamma):
    
        eta = (gamma+1)/2
                
        eta = eta[(abs(eta-1)>0.000001) & (abs(eta)>0.000001)]
        eta.reshape(len(eta),1)
                
        # log is important to make the non-hermitian ssh model second critial 
        # point correct.
        sa = -eta.T @ np.log(abs(eta)) 
        
        # debug
        # sa = -eta.T @ np.log(eta) 
        
        return sa
    
    def get_SCorr_(self,corr):
        
        eta,_ = alg.eig(corr)
        
        eta = eta[(abs(eta-1)>0.000001) & (abs(eta)>0.000001)]
        eta.reshape(len(eta),1)
        
        sa = -eta.T @ np.log(abs(eta)) - (1-eta).T @ np.log(abs(1-eta))
        
        return sa
    
    
    def get_renyi_(self, renyi, gamma):
        " Renyi entropy "
               
        eta = (gamma+1)/2;
        cutoff = 10**(-7);
        
        if self.is_non_herm == 0:
            eta = eta[eta > cutoff]
            eta = eta[(1-eta) > cutoff]
        else:
            eta = eta[abs(eta) > cutoff]
            eta = eta[abs(eta-1) > cutoff]
            
        Rfac = eta*abs(eta)**(renyi-1)+(1-eta)*abs(1-eta)**(renyi-1)
        R = np.log(Rfac.prod())/2
        
        return R
    
    
    " Logarithmic negativity "
    def get_LN_(self,G11,G12,G21,G22):
        
        Gp = np.vstack((np.hstack((-G11, 1j*G12)),
                        np.hstack((1j*G21, G22))))
        Gm = np.vstack((np.hstack((-G11, -1j*G12)),
                        np.hstack((-1j*G21, G22))))
        
        Id = np.eye(Gp.shape[0])
        Gc = Id - (Id-Gm) @ alg.inv(Id+Gp@Gm) @ (Id-Gp)
        
        if self.is_non_herm == 0:
            Gc = 1/2*(Gc+Gc.conj().T)
        
        # this is for first critical point
        # At this point, eig_gamma is real, while eig_gc is not.
        # eig_gc = alg.eig(Gc)[0].real
        # eig_gamma = self.eig_gamma.real
        
        # this is for second critical point
        # At this point, both eig_gamma and eig_gc are not real.
        eig_gc = abs(alg.eig(Gc)[0])
        eig_gamma = self.eig_gamma.real
        
        # test
        eig_gc = (alg.eig(Gc)[0])
        eig_gamma = self.eig_gamma.real
        
        R1 = self.get_renyi_(0.5, eig_gc)
        R2 = self.get_renyi_(2, eig_gamma)
        
        if self.is_non_herm == 0:
            LN = R1+R2/2
        else:
            LN = R1-R2/2
        
    
        return LN

        
    
    " Reflected entropy "
    def get_RE_(self):
        
        if self.is_non_herm == 0:
            GammaR = self.GammaR;
            NA = self.NA;
            
            [Q, T, gamma] = decomp_schur_(1j*GammaR);
            
            isDecomp = abs(Q.T@T@Q - 1j*self.GammaR).sum();
            if isDecomp > 10**(-6):
                print(isDecomp)
                print(" --! [get_RE_] The Schur decomposition is wrong!")
                
            gamma_tilde = abs(1-gamma**2);              
            M_tilde = Q.T @(np.diag(np.sqrt(gamma_tilde)))@Q;           
            MTFD = np.block([[GammaR[0:NA,0:NA],-1j*M_tilde[0:NA,0:NA]],
                             [1j*M_tilde[0:NA,0:NA],-GammaR[0:NA,0:NA]]]);
    
            eig_MTFD,_ = alg.eig(MTFD)
            RE = self.get_S_(eig_MTFD)
         
        elif self.is_non_herm == 1:
            GammaR = self.GammaR;
            #[gamma, R, L] = sort_biortho(GammaR,knum=len(GammaR));
            
            #print("nothing")
            RE = 0;

        return RE;
    
    def get_RECorr_(self):
        
        corr = self.corr;
        D = np.eye(len(corr))-corr.T;
    
        if self.is_non_herm == 0:
            gamma, R = sort_ortho(D)
            
            is_ortho = abs(R.conj().T @ R - np.eye(len(R))).sum();
            if is_ortho > 10**(-6):
                print(is_ortho)
                print(" --! [get_RECorr_] The eigen decomposition is wrong!")
                        
            gamma_tilde = np.diag((lambda x: np.sqrt(abs(x*(1-x))))(abs(gamma)))
            dRef = R @ gamma_tilde @ R.conj().T;
            
        else:
            
            # L.conj().T @ D @ R - np.diag(gamma) = 0
            [gamma, R, L] = sort_biortho(D,knum=len(corr), PT = self.PT)
            # gamma_tilde = np.diag(np.array([my_sqrt_(x*(1-x))*np.sign(x.real) for x in gamma]))

            x_diag = np.zeros(len(gamma),dtype=np.complex128)
            for ix in range(len(gamma)):
                x = gamma[ix]
                if abs(x.imag) < 10**(-8):
                    # When x is real, either positive or negative
                    x_diag[ix] = my_sqrt_(x*(1-x))*np.sign(x.real)
                else:
                    # When x is complex
                    x_diag[ix] = -my_sqrt_(x*(1-x))*np.sign(x.imag)
                    
            gamma_tilde = np.diag(x_diag)
            
            dRef = R @ gamma_tilde @ L.conj().T;
        
            # -----------------------------------------------------------------------------
            # Comment: for the non-Hermitian case, x*(1-x) is negative. 
            # So there is ambiguity over how to choice the branch cut for the square root.
            # Here I make the choice with sign(x.real) such that they come in +- pairs.
            # This choice also reproduces c = -2 for the non-Hermitian SSH model
            # But there is no theoretical proof that this IS the right choice...
            #
            # The result is the same if we use sign(-x.real)
            #-----------------------------------------------------------------------------
        
        NAC = int(self.NA/2);
        DRef = np.block([[D[0:NAC,0:NAC], dRef[0:NAC,0:NAC]],
                         [dRef[0:NAC,0:NAC], np.eye(NAC)-D[0:NAC,0:NAC]]]);

        RE = self.get_SCorr_(DRef);

        return RE;
        
    
    def get_MI_(self,eig_G11,eig_G22):
              
        SA = self.get_S_(eig_G11)
        SB = self.get_S_(eig_G22)
        
        SAB = self.get_S_(self.eig_gamma)
        
        MI = SA+SB-SAB;
        
        return MI;
        
    

#---------------------------------------------------------------------------#

def my_sqrt_(x):   
    if abs(x.imag) < 10**(-9) and x >= 0:
        # When x is real and positive
        return np.sqrt(x)
    
    elif abs(x.imag) < 10**(-9):
        # When x is real and negative
        return np.sqrt(abs(x))*1j
    
    else:
        # When x is complex
        sqrtx = cmath.sqrt(x)
        sqrtx = -sqrtx*np.sign(sqrtx.real)
        
        return sqrtx
    

