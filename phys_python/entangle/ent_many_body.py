# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 16:41:04 2021

@author: Yuhan Liu
"""
import numpy as np

from scipy.sparse import identity as sparse_id
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import scipy.linalg as alg

from toolkit.file_io import File_access
from toolkit.plot_style import plot_style_single
from hamiltonian.spin_tool import get_Sz
from entangle.ent_fit import fit_ent 



def get_ent_many_total(Model, level=0, q=-10, renyi=1, method='biortho', isfit = 1, is_savefig = 0, even_odd = ''):
    
    state_R, state_L, N = Model.R[:,level], Model.L[:,level], Model.N
    #state_R, state_L, N = Model.eigvec[:,level], Model.eigvec[:,level], Model.N
    
    if method=='usual':
        state_R = state_R/np.sqrt(state_R.conj().T @ state_R)
        state_L = state_R
    
    
    if Model.PBC == 0:
        if even_odd == 'odd': [start, incre] = [1, 2]  
        elif even_odd == 'all': [start, incre] = [1, 1]
        else: [start, incre] = [2, 2]  
        
    else:                        
        if even_odd == 'even':  [start, incre] = [2, 2]               
        elif even_odd == 'odd': [start, incre] = [1, 2]
        elif even_odd == 'all': [start, incre] = [1, 1]
        else: [start,incre] = [Model.bands, Model.bands]
           
    
    int_tot = np.array(range(start, Model.N, incre))                           
    ent_tot = np.zeros(len(int_tot))
    
    for i in range(len(int_tot)):
        interval = int_tot[i]
                    
        if interval < N/2+1:
            ent = get_ent_many(state_R, state_L,N,interval,q=q,renyi=renyi)
            print("entanglement entropy is: %f+%fi" % (ent.real,ent.imag))
            ent_tot[i] = ent.real
        else:
            ent_tot[i] = ent_tot[len(int_tot)-1-i]
    
           
    plt.scatter(np.array(int_tot),ent_tot,s=15,color='blue')
    if isfit == 1:
        coeffs, fit_cov, fit_func = fit_ent(int_tot, ent_tot, N, renyi = renyi)
        
        
        x_data = np.arange(int_tot[0],int_tot[-1],(int_tot[-1]-int_tot[0])/50)
        plt.plot(x_data, fit_func(x_data, *coeffs), 'r-',
         label = 'fit: coeff=%5.3f, offset=%5.3f' % tuple(coeffs))
    else:
        plt.plot(np.array(int_tot),ent_tot)
        coeffs = 0
        
    fig = plot_style_single(plt, x_labels = '$L_A$', 
            y_labels = (r'$S_%d$' % renyi))
        
        
    if is_savefig == 1:
        Dir = File_access()
        Dir.save_fig(fig) 
        
    return int_tot, ent_tot, coeffs


#----------------------------------------------------------------------------#
def get_ent_many(vR,vL,N,interval, q=-10, segment='left',renyi=1):
    
    """
    Compute the entanglement entropy in many-body system.
    When q is not -10, we are using the modified trace as in Couvreur 1611.08506 
    """
    
    q_factor = sparse_id(2**(N-interval))
    S_Az = sparse_id(2**interval)
    
    if q!=-10:
        sz = get_Sz(interval,flag=1)
        val_array_Sz = q**(-sz)
        x_array_Sz = range(0,2**(interval),1)
        y_array_Sz = x_array_Sz
        S_Az = csr_matrix((val_array_Sz, (x_array_Sz, y_array_Sz)), shape=(2**interval, 2**interval))
        
        sz = get_Sz(N-interval,flag=1)
        val_array_Sz = q**(sz)
        x_array_Sz = range(0,2**(N-interval),1)
        y_array_Sz = x_array_Sz
        S_Bz = csr_matrix((val_array_Sz, (x_array_Sz, y_array_Sz)), shape=(2**(N-interval), 2**(N-interval)))
        
        q_factor = S_Bz
        # Need to understand why the negative sign is at S_Az instead of S_Bz... Not symmetric...
        
    if segment=='left':
        vR_reshape = vR.reshape(2**(N-interval),2**interval)
        vL_reshape = vL.reshape(2**(N-interval),2**interval)
        
    elif segment=='middle':   
        if interval % 2 !=0: print(" --- check your input!!")
        
        start_point = int(N/2-interval/2)
        sequence = np.hstack((np.array(range(0,start_point)), np.array(range(start_point+interval,N)),np.array(range(start_point,start_point+interval))))
                
        vR_new = np.transpose(vR.reshape(np.full(N,2)), sequence)
        vR_reshape = vR_new.reshape(2**(N-interval),2**interval)
        
        vL_new = np.transpose(vL.reshape(np.full(N,2)), sequence)
        vL_reshape = vL_new.reshape(2**(N-interval),2**interval)
    
    # Generate the redueced density matrix
    rho_reduced = vL_reshape.conj().T @ q_factor @ vR_reshape
    rho_eig,_ = alg.eig(rho_reduced)  
    # TODO: eig or eigh?
    
    rho_eig = rho_eig + 0.0j
    rho_eig = rho_eig[abs(rho_eig)>0.00000001]
    
    
    if renyi==1:
        if q != -10:
            # This also works for the q == -10 case
            S = -np.trace(S_Az @ rho_reduced @ alg.logm(rho_reduced))            
        else:
            # debug
            # S = -rho_eig.T @ np.log(rho_eig)
            
            # As pointed out in 2107.13006, this is equivalent to Couvreur"
            S = -rho_eig.conj().T @ np.log(abs(rho_eig))  
                            
    else:
        # Renyi entropy
        if q != -10:
            S = 1/(1-renyi)*np.log(
                np.trace(S_Az @ np.linalg.matrix_power(rho_reduced,renyi)))     
        else:    
            # debug
            # S = 1/(1-renyi)*np.log((rho_eig**renyi).sum())
                       
            S = 1/(1-renyi)*np.log((rho_eig*abs(rho_eig)**(renyi-1)).sum())
            # TODO: The second renyi entropy for XXZ doesn't give the correct results...
                
    return S

       