"""
Bring matrices to canonical forms. 

Created on Mon Nov  1 12:48:23 2021
@author: Yuhan Liu
"""
import numpy as np
import scipy.linalg as alg
from math import pi
from scipy.sparse import issparse
from scipy.sparse.linalg import eigs as sparse_eigs
from copy import copy
# from scipy.sparse.linalg import eigsh as sparse_eigsh

from toolkit.check import check_zero, check_symmetric, \
    check_diag, check_identity


ERR_CUTOFF = 10**(-8)
ERR_CUTOFF_LOOSE = 10**(-4)


def sort_ortho(h, knum=-1,eig_which='SR'):
    """
    Obtain the eigensystem of hermitian Hamiltonian

    Parameters
    ----------
    hamiltonian : numpy array
        Hermitian Hamiltonian

    Returns
    -------
    eigval : numpy array
    eigvec : numpy array (matrix)
        Every column is an eigenvector.

    """
    n_h = h.shape[0]
    
    if knum > 0 and knum < n_h:
        eigval, eigvec = sparse_eigs(h, k=knum, which=eig_which)
    else:
        eigval, eigvec = alg.eigh(h)
    
    eigval = eigval.real
    V_norm = np.diag(1/np.sqrt(np.diag(eigvec.conj().T @ eigvec)))
    eigvec = eigvec @ V_norm
    
    eigval, eigvec = sort_real(eigval, eigvec)
    
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
                      
            regVR = sort_block_(regVR)
            eigvec[:,reg] = regVR
                    
    is_show = 1
    identity_error = check_identity(eigvec.conj().T @ eigvec, is_show)
    if identity_error > 10**(-6): print(
            " --> [sort_ortho] Orthonormal error: %f" % identity_error)
    
    zero_error = check_zero(eigvec.conj().T @ h @ eigvec - np.diag(eigval))
    if zero_error > 10**(-6): print(
            " --> [sort_ortho] Decomposition error: %f" % zero_error) 
    
    return eigval, eigvec


def sort_block_(regV):
    reg_num = (regV.shape)[1]
    
    overlap = regV.conj().T @ regV
    if np.sum(abs(overlap - np.identity(reg_num)))>10**(-7):
        
        print("Sort manually...")
        eig_ov, vec_ov = alg.eigh(overlap)
        regV = regV @ vec_ov
        vr_norm = np.diag(1/np.sqrt(np.diag(regV.conj().T @ regV)))
        regV = regV @ vr_norm

    return regV





def sort_biortho(h,knum = -1, eig_which='SR', PT='true', is_jordan = 0,
                 is_remove_zero = 0):
    
    # knum is only used for large system
    """
    #--------------------------------------------------------------------------#
    # COMMENT:
    # 1. If H is symmetric, for H|R> = E|R>, H^\dag |L> = E^* |L>, we have:
    #   |L> = |R>^*. Here, PT actually means H is symmetric. 
    #
    # 2. Here PT = 'true' means both the Hamiltonian and the eigenstates preserve
    #   the PT symmetry. This guarantees all the eigenvalues are real.
    #   (Seems like it does not matter in numerics...)
    #--------------------------------------------------------------------------#

    """
        
    
    n_h = h.shape[0]
    
    if knum > 0 and knum < n_h:
        evl, ev = sparse_eigs(h, k=knum, which=eig_which)
    else:    
        if issparse(h):
            h = h.todense()
            
        evl, ev = alg.eig(h)
        
    evl, ev = sort_real(evl, ev)


    if PT == 'true' and check_symmetric(h):
        ev_L = ev.conj()
      
    else:
        if PT == 'true':
            print(" [sort_biortho] Error: Input matrix is not symmetric!")
            PT = 'false'

        if knum > 0:
            evl_L, ev_L = sparse_eigs(h.conj().T, k=knum, which=eig_which)
        else:
            evl_L, ev_L = alg.eig(h.conj().T)
            
        evl_L, ev_L = sort_real(evl_L, ev_L)
     
        
    "Remove the elements that are too close to 0 or 1, in the reflected"
    "entropy calculation"
    if is_remove_zero == 1:
        idx = np.where((abs(evl)>10**(-10)) & (abs(evl-1)>10**(-10)))[0]
        evl = evl[idx]
        ev = ev[:, idx]
        ev_L = ev_L[:,idx]

         
    labels=[-1]
    R = ev+np.zeros(ev.shape,dtype=complex)
    L = ev_L+np.zeros(ev.shape,dtype=complex)
    
    for i in range(len(evl)-1):
        # if abs(evl[i+1]-evl[i]) > ERR_CUTOFF:
        if abs(evl[i+1])-abs(evl[i]) > ERR_CUTOFF:
            labels.append(i)
            
    if (labels.count(len(evl)-1) == 0):
        labels.append(len(evl)-1)

       
    # TODO: it seems the result is sensitive to the ERR_CUTOFF in the label 
    #       computation above...
    for i in range(len(labels)-1):
        if labels[i+1]-labels[i]>1:
            reg = range(labels[i]+1,labels[i+1]+1)
            regVR = ev[:,reg] 
            regVL = ev_L[:,reg]
                     
            if PT == 'true':
                R[:,reg] = __sort_block(regVR, reg)
                L[:, reg] = R[:, reg].conj()
                
            else:
                # regVL  = __sort_block(regVL, reg)
                R[:, reg], L[:,reg] = __sort_general(regVR, regVL)
                        
    # TODO: we cannot normalize the Jordan block
    
    if is_jordan == 0:
        V_norm = np.diag(1/np.sqrt(np.diag(L.conj().T@R)))
        R = R @ V_norm  
        L = L @ V_norm.conj()        

    is_show = 0
    err_diag = check_diag(L.conj().T @ R,is_show)
    err_h = check_diag(abs(L.conj().T@ h @R),is_show)
    if err_diag > ERR_CUTOFF_LOOSE:
        print(" [sort_biortho] error for orthonormal: %f" %  err_diag)
    if err_h > ERR_CUTOFF_LOOSE:
        print(" [sort_biortho] error for H: %f" % err_h)
    

    return evl, R, L


def __sort_general(VR, VL):
    
    # Deal with degeneracy in the case where h is not symmetric.
    # Use the idea of echolon decomposition.
    
    N = len(VR.T)
    A = VL.conj().T @ VR
    A = A.T
    
    VRtr = copy(VR).T
    
    j = 0
    while j<N:
        for i in range(j+1,N):
            if abs(A[i,j]) < ERR_CUTOFF:
                pass
            elif abs(A[j,j]) < ERR_CUTOFF:
                A[[i,j]] = A[[j,i]]
                VRtr[[i,j]] = VRtr[[j,i]]
            else:
                fac = (A[i, j]/A[j,j])
                A[i,:] = A[i,:] - A[j,:]*fac
                A[i,j] = 0
                
                VRtr[i,:] = VRtr[i,:] - VRtr[j,:]*fac
                
        j = j+1
    
    j = 1
    while j<N:
        # TODO: this doesn't deal with general case
        
        for i in range(0,j):
            if abs(A[i,j]) < ERR_CUTOFF:
                pass
            else:
                fac = (A[i,j]/A[j,j])
                A[i,:] = A[i,:] - A[j,:]*fac
                A[i,j] = 0
                
                VRtr[i,:] = VRtr[i,:] - VRtr[j,:]*fac
            
        j = j+1
        
    VRtr = VRtr.T
    
    """
    Note: if the diagonal element of VL.conj().T @ VRtr is too small, the error
    can be huge later on.
    """
    # test = VL.conj().T @ VRtr
    # print(check_diag(test))
    # print("min", min(abs(np.diag(test))))
        
    return VRtr, VL


def __sort_block(regVR, reg):
    
    if np.sum(abs(regVR.T@regVR-np.identity(len(reg))))>10**(-7):
        
        V_unnorm = __Takagifac(regVR)
        eig_fac = np.diag(1/np.sqrt(np.diag(V_unnorm.T@V_unnorm)))               
        
        V_norm = V_unnorm@eig_fac
        overlap = V_norm.T @ V_norm
        
        check_diag(overlap)
        
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
    else:
        V_norm = regVR
             
    return V_norm


def __Takagifac(R):
    # Autonne-Takagi factorization
    # D = UAU^T where A is a complex symmetric matrix, U is a unitary. D is real non-negative matrix
    
    # https://en.wikipedia.org/wiki/Symmetric_matrix#Complex_symmetric_matrices

    
    A = R.T @ R
    
    if (abs(A-np.diag(np.diag(A))).sum()) > 10**(-6): 
        
        _,V = alg.eigh(A.conj().T @ A)        
        C = V.T @ A @ V

        if (abs(C-np.diag(np.diag(C))).sum()) > 10**(-6):   
            _,W = alg.eigh((V.T @ A @ V).real)
            U = W.T @ V.T
        else:
            U = V.T
            
        Up = np.diag(np.exp(-1j*np.angle(np.diag(U @ A @ U.T))/2)) @ U    
        
        R = R@Up.T
    
    return R



def sort_P_nonh(E,V,P,H,N):
    
    # TODO: there is a duplicate of this in spin_tool

    R = V+np.zeros(V.shape,dtype=complex)
    L = R.conj()
    # Need to specify V is complex. Otherwise it will take real part of V[:,reg]=regV@Vtrans
    labels=[-1]
    for i in range(len(E)-1):
        if (E[i+1]-E[i]).real>0.0000001:
            labels.append(i)
            
    for i in range(len(labels)-1):
        if labels[i+1]-labels[i]>1:
            reg=range(labels[i]+1,labels[i+1]+1)
            regV=V[:,reg]
            Peig=regV.T @ P @ regV
            # Sometimes, S obtained from the eigenvalue of Peig is not integer... 
            # This may because of our numerical way to get the eigensystem. 
            # Some states might be failed to be included.
            
            # Peig is not necessarily hermitian! Using eig might not be safe? 
            # I guess it is still safe because Peig = V*P_{diag}*V^{-1} is still valid
            
            S,Vtrans=alg.eig(Peig)
            R[:,reg] = regV@Vtrans
            L[:,reg] = regV.conj()@(alg.inv(Vtrans)).conj().T

            
            # After this, L is not necessary the conjugate of R
    
    P_eig = L.conj().T @ P @ R
    S = np.angle(P_eig.diagonal())*N/(2*pi)
    print("error for orthonormal: %f" % 
      check_diag(L.conj().T @ R, is_show = 1))
    print("error for H: %f" % 
      check_diag(L.conj().T @ H @ R, is_show = 1))
    print("error for P: %f" % 
      check_diag(P_eig, is_show = 1))
               
    
    return R, L, S



def simult_diag_nonh(H, M, knum = -1):
    # TODO: doesn't work yet
    
    if isinstance(M, np.ndarray): M = [M]
    if issparse(M): M = [M]
    
    
    if issparse(H)==1:
        N_dimH = H.get_shape()[0]
        N_chain = int(np.log2(N_dimH))
    else:
        N_chain = len(H)
        
        
    ep = 10**(-4)
    ep2 = 10**(-5)
    ep3 = 10**(-6)
    if len(M) == 1:
        Hp = H + ep*M[0]
    elif len(M) == 2:
        Hp = H + ep*M[0] + ep2*M[1]
    elif len(M) == 3:
        Hp = H + ep*M[0] + ep2*M[1] + ep3 * M[2]
    
    if knum > 0:
        eigval_R, eigvecs_R = sparse_eigs(Hp, k=knum, which='SR')
        eigval_L, eigvecs_L = sparse_eigs(Hp.conj().T, k=knum, which='SR')
    else:
        eigval_R, eigvecs_R = alg.eig(Hp)
        eigval_L, eigvecs_L = alg.eig(Hp.conj().T)
    
    eigval_R, eigvecs_R = sort_real(eigval_R, eigvecs_R)
    eigval_L, eigvecs_L = sort_real(eigval_L, eigvecs_L)
    
    norm_sqrt = np.diag(eigvecs_L.conj().T @ eigvecs_R)
    eigvecs_R = eigvecs_R @ np.diag(1/np.sqrt(norm_sqrt))
    eigvecs_L = eigvecs_L @ np.diag(1/np.sqrt(norm_sqrt)).conj()
    
    return eigval_R, eigvecs_R


def simult_diag(H, M, knum = -1, which = 'SR', is_phase = 0, is_show = 0, is_sort = 1, 
                bands = 1, is_zero_sym = 1, N_chain = None):
    
    """
    simult_diag(H,M) or simult_diag(H,[M0, M1])
    
    Simultaneously diagonalize H and M. (or H, M[0], M[1])

    Although Hp is not hermitian, for some reason, the eigvecs are still 
    orthogonal. The error is very small, as long as there is no further
    degeneracy. 
    
    TODO: "is_zero_sym" part only works for two zero states
    """
    
    if isinstance(M, np.ndarray): M = [M]
    if issparse(M): M = [M]

    
    if issparse(H)==1:
        N_dimH = H.get_shape()[0]
        if N_chain == None: N_chain = int(np.log2(N_dimH))
    else:
        if N_chain == None: N_chain = len(H)
    
    
    ep = 10**(-4)
    ep2 = 10**(-5)
    if len(M) == 1:
        Hp = H + ep*M[0]
    elif len(M) == 2:
        Hp = H + ep*M[0] + ep2*M[1]
    
    if knum > 0:
        eigval, eigvecs = sparse_eigs(Hp, k=knum, which=which)
    else:
        eigval, eigvecs = alg.eig(Hp)
        
    
    if is_sort == 1: 
        eigval, eigvecs = sort_real(eigval, eigvecs)
        
    
    eig_H = eigvecs.conj().T@ H @ eigvecs
    eig_M0 = eigvecs.conj().T@ M[0] @ eigvecs
    if len(M) == 2:
        eig_M1 = eigvecs.conj().T@ M[1] @ eigvecs
    
    
    # Resolve the residual degeneracy manually...
    label = []
    if len(M) == 1:
        for i in range(len(eig_H)-1):
            if abs(eig_H[i,i]-eig_H[i+1,i+1])<ERR_CUTOFF \
                and abs(eig_M0[i,i]-eig_M0[i+1,i+1])<ERR_CUTOFF:
                    label.append(i)
    elif len(M) == 2:
        for i in range(len(eig_H)-1):
            if abs(eig_H[i,i]-eig_H[i+1,i+1])<ERR_CUTOFF \
                and abs(eig_M0[i,i]-eig_M0[i+1,i+1])<ERR_CUTOFF\
                    and abs(eig_M1[i,i]-eig_M1[i+1,i+1])<ERR_CUTOFF:
                    label.append(i)
        
   
    if bool(label):
        start = []
        end = []
        
        for ele in label:
            if not ele-1 in label: start.append(ele)
            if not ele+1 in label: end.append(ele+1)
        
        for i in range(len(start)):
            reg = range(start[i], end[i]+1)
            regV = eigvecs[:,reg]
            regV = sort_block_(regV)
            
            eigvecs[:,reg] = regV
            
    if bool(label):
        eig_H = eigvecs.conj().T@ H @ eigvecs
        eig_M0 = eigvecs.conj().T@ M[0] @ eigvecs
       
    print(" [simult_diag] error for orthonormal: %f" 
          % check_diag(eigvecs.conj().T @ eigvecs, is_show = is_show))
    
    print(" [simult_diag] error for H: %f" 
          % check_diag(eig_H, is_show = is_show))
        
    print(" [simult_diag] error for M[0]: %f" 
          % check_diag(eig_M0, is_show = is_show))
    
    if len(M) == 2:
        eig_M1 = eigvecs.conj().T@ M[1] @ eigvecs
        print(" [simult_diag] error for M[1]: %f" 
              % check_diag(eig_M1, is_show = is_show))
    
    

    if is_phase == 1:
        # should be integers
        eig_M0 = np.angle(np.diag(eig_M0))*N_chain/(2*pi)
        eig_M0 = eig_M0/bands
        
        l_brillouin = N_chain/bands
        
        left = -l_brillouin/2 - 10**(-4)
        right = l_brillouin/2 + 10**(-4)
    
        loc_l_out = np.where(eig_M0<left)[0]
        eig_M0[loc_l_out] = eig_M0[loc_l_out] + l_brillouin
        loc_r_out = np.where(eig_M0>right)[0]
        eig_M0[loc_r_out] = eig_M0[loc_r_out] - l_brillouin
        
    else:
        eig_M0 = np.diag(eig_M0)
        
        
    if is_zero_sym == 1:
        eig_Hd = np.diag(eig_H).real
        zero_loc = np.where(abs(eig_Hd)<10**(-8))[0]
        flip_mtr = np.eye(len(eig_H))
        
        if len(zero_loc)>1:
            if (eig_Hd[zero_loc[0]]>0 and eig_Hd[zero_loc[1]]>0) or\
                (eig_Hd[zero_loc[0]]<0 and eig_Hd[zero_loc[1]]<0):
                
                flip_mtr[zero_loc[0],zero_loc[0]] = -1
                eig_H = eig_H @ flip_mtr
       
    eig_H = np.diag(eig_H).real
    
       
    if len(M) == 1: eig_M = eig_M0
    elif len(M) == 2: eig_M = [eig_M0, np.diag(eig_M1)]
    
    return eig_H, eigvecs, eig_M
    
    
    

def simult_diag_old(H, E, V, M, is_quiet = 1, is_quiet_debug = 1, is_phase = 0):
    
    # Please use simult_diag.
    # This is just for record
    
    if isinstance(M, np.ndarray): M = [M]
    if issparse(M): M = [M]
    
    N_eig = len(E)
    
    if issparse(H)==1:
        N_dimH = H.get_shape()[0]
        N_chain = int(np.log2(N_dimH))
    else:
        N_chain = len(H)
    
    test = V.conj().T@ H @ V
    if is_quiet == 0:
        print(" [simult_diag] before sort: error for orthonormal: %f" 
              % check_diag(V.conj().T @ V))
        print(" [simult_diag] before sort: error for H: %f" 
              % check_diag(test))
    
    V_sort = V+np.zeros(V.shape,dtype=complex)
    # Need to specify V_sort is complex. Otherwise it will take real part 
    # of V_sort[:,reg]=regV@Vtrans
    labels=[-1]
    for i in range(len(E)-1):
        if E[i+1]-E[i]>0.0000001:
            labels.append(i)
            
    if labels[-1] != N_eig-1: labels.append(N_eig-1)
            
    for i in range(len(labels)-1):
        if labels[i+1]-labels[i]>1:
            reg = range(labels[i]+1,labels[i+1]+1)
            regV = V[:,reg]
            Meig = regV.conj().T@ M[0] @regV
            
            # Meig is not necessarily hermitian! So we use eig instead of eigh
            # TODO: Vtrans is not guaranteed to be orthonormal...
            #       This is because of the degeneracy in S. 
            # Comment:  even that Peig is not hermitian, if there is no degeneracy
            #           in S, then Vtrans is forced to be orthogonal. This 
            #           is because eigenvectors with different eigenvalues
            #           cannot mix.
            S,Vtrans=alg.eig(Meig)
            
            # Do not transform if it is already diagonal
            if check_diag(Meig)<10**(-6): Vtrans = np.eye(len(Meig))
            
            V_sort_local = regV@Vtrans
                       
            if len(M) == 2:
                _, V_sort_local = simult_diag(H, E[reg], V_sort_local, M[1])
            
            if is_quiet_debug == 0: 
                # The following lines check the orthogonality
                vtest = V_sort_local
                # print(alg.norm(vtest,axis=0))
                test = Vtrans.conj().T @ Vtrans 
                print("error for orthonormal: %f" % check_diag(test))
                print("error for H: %f" % 
                      check_diag(vtest.conj().T@ H @ vtest))
                print("error for P: %f" % check_diag(
                    vtest.conj().T@ M[0] @ vtest))
            
            
            V_sort[:,reg] = V_sort_local
            

    eig_M = V_sort.conj().T @ M[0] @ V_sort
    # S = np.angle(eig_P.diagonal())*self.N/(2*pi)
    
    if is_quiet == 0:
        print(" [simult_diag] error for orthonormal: %f" 
              % check_diag(V_sort.conj().T @ V_sort))
        print(" [simult_diag] error for H: %f" 
              % check_diag(V_sort.conj().T@ H @V_sort))
        print(" [simult_diag] error for P: %f" 
              % check_diag(eig_M))
    
    
    if is_phase == 1:
        # should be integers
        eig_M = np.angle(np.diag(eig_M))*N_chain/(2*pi)
        
        left = -N_chain/2 - 10**(-4)
        right = N_chain/2 + 10**(-4)
    
        loc_l_out = np.where(eig_M<left)[0]
        eig_M[loc_l_out] = eig_M[loc_l_out] + N_chain
        loc_r_out = np.where(eig_M>right)[0]
        eig_M[loc_r_out] = eig_M[loc_r_out] - N_chain
            
    return eig_M, V_sort


def sort_real(eigval, eigvecs, left = 0):
    if left == 0:
        idx = eigval.argsort()[::1] # seems this sort both the real and the imag   
    elif left == 1:
        idx = eigval.conj().argsort()[::1]
    
    eigval = eigval[idx]
    eigvecs = eigvecs[:,idx]

    return eigval, eigvecs


def move_brillouin(S, N_brillouin):
    
    left = -N_brillouin/2 - 10**(-4)
    right = N_brillouin/2 + 10**(-4)

    loc_l_out = np.where(S<left)[0]
    S[loc_l_out] = S[loc_l_out] + N_brillouin
    loc_r_out = np.where(S>right)[0]
    S[loc_r_out] = S[loc_r_out] - N_brillouin
    
    return S
    

def fold_brillouin(S, N):
    """
    Fold the "Brillouin zone" for antiferromagetic case. 

    Returns
    -------
    S : numpy.array

    """
    
    N_half = N/2
    N_quad = N/4
    
    for i, s in enumerate(S):
        if s>N_quad: S[i] = s-N_half
        elif s<-N_quad: S[i] = s+N_half
    
    return S


def decomp_schur_(K, is_pure_imag = 0, is_reverse=0):
    """
    Schur decomposition for real anti-symmetric matrix.
    ---------------------------------------------------------------
    K = Q.'*T*Q
    - input: K is a real anti-symmetric
    - output: 1. Q is an orthogonal matrix.
              2. T is block diagonal matrix [0, lambda; -lambda, 0],
              where lamda is non-negative.
              3. Lambda = [lambda_1,lambda_1,lambda_2,lambda_2,...]
              
    If is_pure_imag == 1, the output T is [0, iLambda; -iLambda, 0],
    output Lambda is iLambda.
    
    If is_reverse == 1, the output T is [0, -lambda; lambda, 0]
    ------------------
    return Q, T, Lambda
              
    """
    if is_pure_imag == 1: K = K*(-1j)
    
    
    isReal = abs(K.imag).sum()
    if isReal > 10**(-6):
        print(isReal)
        print(" --! [decomp_schur_] The K-matrix is not REAL!")
        
    K = K.real;
    
    isSkewSym = abs(K+K.conj().T).sum();
    if isSkewSym > 10**(-6):
        print(isSkewSym)
        print(" --! [decomp_schur_] The K-matrix is not anti-symmetric!")
    
    K = (K-K.conj().T)/2;
    
    # Q.T @ K @ Q - T = 0
    T, Q = alg.schur(K, output='real');
    
    QLen = len(Q);
    
    M = np.eye(QLen)*(1+0*1j);
    Lambda = np.zeros(QLen);
    
    for i in range(int(QLen/2)):
        
        if is_reverse == 0:
            if T[2*i,2*i+1] < T[2*i+1,2*i]:
                M[2*i:2*i+2,2*i:2*i+2] = np.array([[0,1],[1,0]])
        else:
            if T[2*i,2*i+1] > T[2*i+1,2*i]:
                M[2*i:2*i+2,2*i:2*i+2] = np.array([[0,1],[1,0]])
        
        
        Lambdai = abs(T[2*i,2*i+1]);
#        if abs(Lambdai-1)<10^(10):
#            Lambdai = 1;
        
        Lambda[2*i] = Lambdai;
        Lambda[2*i+1] = Lambdai;
    
    T = M@T@M;
    Q = Q@M;
    
    # such that K = Q.'*T*Q
    Q = Q.T;
    
    if is_pure_imag == 1: 
        T = T*1j
        Lambda = Lambda*1j
    
    return Q, T, Lambda



def takagi_decomp_(A):
    """
    Autonne-Takagi factorization for complex symmetric matrix. 
    https://en.wikipedia.org/wiki/Symmetric_matrix#Complex_symmetric_matrices
    A = U^T @ D @ U
    
    Parameter
    ----------
    A : complex symmetric matrix.

    Returns
    -------
    U : unitary matrix
    D : real non-negative matrix

    """

    if not check_symmetric(A):
        print(" --! [takagi_decomp_] Input is not symmetric matrix!")
        return 0, 0

    _, V = alg.eigh(A.conj().T @ A)
    _, W = alg.eigh((V.T @ A @ V).real)
    U = W.T @ V.T
    Up = np.diag(np.exp(-1j*np.angle(np.diag(U @ A @ U.T))/2)) @ U   
    Up = Up.conj()
    
    D = Up.conj() @ A @ Up.conj().T
    
    diag_error = check_diag(D)
    if diag_error > ERR_CUTOFF:
        print(" --> [takagi_decomp_] D diag error: " + str(diag_error))   
        
    unitary_error = check_zero(Up @ Up.conj().T - np.eye(len(Up)))
    if unitary_error > ERR_CUTOFF:
        print(" --> [takagi_decomp_] U unitary error: " + str(unitary_error))
        
    decomp_error = check_zero(A - Up.T @ D @ Up)
    if decomp_error > ERR_CUTOFF:
        print(" --> [takagi_decomp_] Decomposition error: " + str(decomp_error))
    
    return D, Up



if __name__ == "__main__":
    
    case = 2
    
    if case == 1:
        K = np.random.rand(4,4)
        K = K-K.T
        [Q,T,Lambda] = decomp_schur_(K)
        print(check_zero(Q.T @ T @ Q - K))
        
    elif case == 2:
        K = np.random.rand(4,4) + 1j*np.random.rand(4,4)
        K = K+K.T
        [D, U] = takagi_decomp_(K)
        
        
        
    