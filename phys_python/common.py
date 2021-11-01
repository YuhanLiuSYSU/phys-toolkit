"""
Created on Mon Sep 27 20:27:00 2021

Common codes

@author: Yuhan Liu
"""

# from scipy.sparse.linalg import eigs
import numpy as np
import scipy.linalg as alg
import matplotlib.pyplot as plt
from math import pi
from scipy.optimize import curve_fit
import os
import pickle

"""
 - class File_access

 - sort_ortho
 - sort_biortho
 
 - class FitClass

"""

#%%
class File_access:
    # path to save the data
    def __init__(self, use_default = 1):
        
        if use_default == 1:
            my_dir = os.getcwd()
        else:
            # this is the old old path
            my_dir = ("C:\\Users\\sglil\\OneDrive\\Desktop" + 
                        "\\CS\\python\\spin-chain\\spinIsing\\")
            
        self.save_dir = os.path.join(my_dir,'save_results\\')
        if os.path.exists(self.save_dir)==False: os.makedirs(self.save_dir)
         
        
    def save_data(self, result_data): 
        # result_data can be a tuple, like [result, N]
        
        save_file_name = input('--> Input the save data name: '+
                               '(press <ENTER> for not to save)')
        if save_file_name!="":
            with open(self.save_dir+save_file_name+'.pkl', 'wb') as f: 
                pickle.dump(result_data, f)
                
    def save_fig(self,fig):
        save_name = input('--- Input the save fig name (press <ENTER> for not to save): ')
        if save_name!="":
            fig.savefig(self.save_dir+save_name+'.pdf', bbox_inches='tight')
        
        
    def get_back_ext(self,is_from_new):
        if is_from_new == 1:
            open_file_name = input('--- Input the open data name: (press <ENTER> for not to open)')
            with open(self.save_dir+'last_open'+'.pkl', 'wb') as f: 
                pickle.dump(open_file_name,f)
        else:
            with open(self.save_dir+'last_open'+'.pkl','rb') as f:         
                open_file_name = pickle.load(f)
                
        
        return self.get_back(open_file_name)
        
    
    def get_back(self,file_name):  
        with open(self.save_dir+file_name+'.pkl','rb') as f:         
            return pickle.load(f)
        
    def re_save(self,Model):
        with open(self.save_dir+'last_open'+'.pkl', 'rb') as f: 
            open_file_name = pickle.load(f)
            with open(self.save_dir+open_file_name+'.pkl','wb') as f_re:      
                pickle.dump(Model,f_re)


#%%

def sort_ortho(hamiltonian):
    # for hermitian matrix
    
    eigval, eigvec = alg.eigh(hamiltonian)
    
    eigval = eigval.real
    V_norm = np.diag(1/np.sqrt(np.diag(eigvec.conj().T @ eigvec)))
    eigvec = eigvec @ V_norm
    
    idx = eigval.argsort()[::1]   
    eigval = eigval[idx]
    eigvec = eigvec[:,idx]
    
    # deal with degeneracy
    labels=[-1]
    for i in range(len(eigval)-1):
        if abs(eigval[i+1]-eigval[i])>10**(-7):
            labels.append(i)
            
    if (labels.count(len(eigval)-1) == 0):
        labels.append(len(eigval)-1)
    
    # sort each block
    for i in range(len(labels)-1):
        if labels[i+1]-labels[i]>1:
            reg = range(labels[i]+1,labels[i+1]+1)
            regVR = eigvec[:,reg] 
                      
            overlap = regVR.conj().T @ regVR
            if np.sum(abs(overlap - np.identity(len(reg))))>10**(-7):
                eig_ov, vec_ov = alg.eigh(overlap)
                regVR = regVR @ vec_ov
                vr_norm = np.diag(1/np.sqrt(np.diag(regVR.conj().T @ regVR)))
                regVR = regVR @ vr_norm
                
                eigvec[:,reg] = regVR
                    
    is_show = 1
    print(" --> error for identity: %f" % 
      check_identity(eigvec.conj().T @ eigvec, is_show))
    
    return eigval, eigvec
    
#%%
def sort_biortho(hamiltonian,knum = -1, eig_which='SR', PT='true'):
    
    # knum is only used for large system
    
    #--------------------------------------------------------------------------#
    # COMMENT:
    # 1. If H is symmetric, for H|R> = E|R>, H^\dag |L> = E^* |L>, we have:
    #   |L> = |R>^*
    #
    # 2. Here PT = 'true' means both the Hamiltonian and the eigenstates preserve
    #   the PT symmetry. This guarantees all the eigenvalues are real.
    #   (Seems like it does not matter in numerics...)
    #--------------------------------------------------------------------------#
    
    #eigval, eigvecs = eigs(hamiltonian, k=knum, which=eig_which)
    eigval, eigvecs = alg.eig(hamiltonian)
    idx = eigval.argsort()[::1]   
    eigval = eigval[idx]
    eigvecs = eigvecs[:,idx]
    
    if PT!='true':
        #eigval_L, eigvecs_L = eigs(hamiltonian.conj().T, k=knum, which=eig_which)
        eigval_L, eigvecs_L = alg.eig(hamiltonian.conj().T)
        idx = eigval_L.argsort()[::1]   
        eigval_L = eigval_L[idx]
        eigvecs_L = eigvecs_L[:,idx]
    
    
    V_norm = np.diag(1/np.sqrt(np.diag(eigvecs.conj().T@eigvecs)))
    eigvecs = eigvecs@V_norm
    
    labels=[-1]
    eigvecs_sort = eigvecs+np.zeros(eigvecs.shape,dtype=complex)
    for i in range(len(eigval)-1):
        if abs(eigval[i+1]-eigval[i])>10**(-7):
            labels.append(i)
            
    if (labels.count(len(eigval)-1) == 0):
        labels.append(len(eigval)-1)
        
 #       print("debug")
        
    for i in range(len(labels)-1):
        if labels[i+1]-labels[i]>1:
            reg = range(labels[i]+1,labels[i+1]+1)
            regVR = eigvecs[:,reg] 
            
            
            if np.sum(abs(regVR.T@regVR-np.identity(len(reg))))>10**(-7):
                
                V_unnorm = __Takagifac(regVR)
                eig_fac = np.diag(1/np.sqrt(np.diag(V_unnorm.T@V_unnorm)))               
                
                V_norm = V_unnorm@eig_fac
                overlap = V_norm.T @ V_norm
                
                check_diag(overlap)
                # tsave = V_norm[:,:]
                
                subreg = []
                for j in range(len(reg)-1):
                    # Sort again
                    
                    if abs(overlap[j,j+1])>0.000001:
                        subreg.extend([j,j+1])
                subreg = list(set(subreg))
                if subreg!=[]:
                    
                    subreg_VR = V_norm[:,subreg]
                    V_unnorm_2 = __Takagifac(subreg_VR)
                    eig_fac_2 = np.diag(1/np.sqrt(np.diag(V_unnorm_2.T@V_unnorm_2)))   
                    V_norm_22 = V_unnorm_2@eig_fac_2
                    V_norm[:,subreg] = V_norm_22
                    
                    # test4 = test
                    # test4[:,subreg] = V_norm_22
                    # test3 = test4.T @ test4
                    # plt.imshow(abs(test3), cmap = 'jet')
                    # plt.colorbar()
                                          
                eigvecs_sort[:,reg] = V_norm
                
    V_norm = np.diag(1/np.sqrt(np.diag(eigvecs_sort.T@eigvecs_sort)))
    eigvecs_sort = eigvecs_sort@V_norm        

    is_show = 0
    print(" --> error for orthonormal: %f" % 
      check_diag(eigvecs_sort.T @ eigvecs_sort,is_show))
    print(" --> error for H: %f" % 
      check_diag(abs(eigvecs_sort.T@ hamiltonian @eigvecs_sort),is_show))
    
    R = eigvecs_sort
    L = eigvecs_sort.conj()
    
    return eigval, R, L


#def __Takagifac(L,R):
## Autonne-Takagi factorization
## D = UAU^T where A is a complex symmetric matrix, U is a unitary. D is real non-negative matrix
#
#    A = L.conj().T @ R 
#    _,V = alg.eig(A.conj().T @ A)
#    _,W = alg.eig((V.T @ A @ V).real)
#    U = W.T @ V.T
#    Up = np.diag(np.exp(-1j*np.angle(np.diag(U @ A @ U.T))/2)) @ U    
#    
#    return L@Up.conj().T, R@Up.T

def __Takagifac(R):
    # Autonne-Takagi factorization
    # D = UAU^T where A is a complex symmetric matrix, U is a unitary. D is real non-negative matrix
    
    # https://en.wikipedia.org/wiki/Symmetric_matrix#Complex_symmetric_matrices

    
    A = R.T @ R
    
    if (abs(A-np.diag(np.diag(A))).sum()) > 10**(-6): 
        
        _,V = alg.eigh(A.conj().T @ A)        
        C = V.T @ A @ V

        if (abs(C-np.diag(np.diag(C))).sum()) > 10**(-6):   
            # TODO: eig or eigh?
            _,W = alg.eig((V.T @ A @ V).real)
            U = W.T @ V.T
        else:
            U = V.T
            
        Up = np.diag(np.exp(-1j*np.angle(np.diag(U @ A @ U.T))/2)) @ U    
        
        R = R@Up.T
    
    return R

#%%
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

#-----------------------------------------------------------------------------#
#%% 
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


def fit_ent(interval, ent, N, renyi = 1, fit_type = 1):
    Fit_func = FitClass()
    Fit_func.renyi = renyi
    Fit_func.N = N
    
    if fit_type == 1:
        my_func = Fit_func.fit_func_ent
    elif fit_type == 2:
        my_func = Fit_func.fit_func_MI    
    elif fit_type == 3:
        my_func = Fit_func.fit_func_LN
    
    
    coeffs, coeffs_cov = curve_fit(my_func,interval,ent)
    
    return coeffs, coeffs_cov, my_func