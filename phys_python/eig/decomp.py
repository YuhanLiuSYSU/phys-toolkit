
"""
Bring matrices to canonical forms. 

Created on Mon Nov  1 12:48:23 2021
@author: Yuhan Liu
"""
import numpy as np
import scipy.linalg as alg

from toolkit.check import check_zero, check_symmetric, \
    check_diag, check_identity


ERROR_CUTOFF = 10**(-6)


def sort_ortho(hamiltonian):
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
    identity_error = check_identity(eigvec.conj().T @ eigvec, is_show)
    if identity_error > 10**(-6): print(
            " --> [sort_ortho] Orthonormal error: %f" % identity_error)
    
    zero_error = check_zero(eigvec.conj().T @ hamiltonian @ eigvec - np.diag(eigval))
    if zero_error > 10**(-6): print(
            " --> [sort_ortho] Decomposition error: %f" % zero_error) 
    
    return eigval, eigvec



def sort_biortho(hamiltonian,knum = -1, eig_which='SR', PT='true'):
    
    # knum is only used for large system
    """
    #--------------------------------------------------------------------------#
    # COMMENT:
    # 1. If H is symmetric, for H|R> = E|R>, H^\dag |L> = E^* |L>, we have:
    #   |L> = |R>^*
    #
    # 2. Here PT = 'true' means both the Hamiltonian and the eigenstates preserve
    #   the PT symmetry. This guarantees all the eigenvalues are real.
    #   (Seems like it does not matter in numerics...)
    #--------------------------------------------------------------------------#
    """
    
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




def decomp_schur_(K):
    """
    Schur decomposition for real anti-symmetric matrix.
    ---------------------------------------------------------------
    K = Q.'*T*Q
    - input: K is a real anti-symmetric
    - output: 1. Q is an orthogonal matrix.
              2. T is block diagonal matrix [0, lambda; -lambda, 0],
              where lamda is non-negative.
              3. Lambda = [lambda_1,lambda_1,lambda_2,lambda_2,...]
              
    """
    
    isReal = K.real.sum();
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
        if T[2*i,2*i+1] < T[2*i+1,2*i]:
            M[2*i:2*i+2,2*i:2*i+2] = np.array([[0,1],[1,0]]);
        
        
        Lambdai = abs(T[2*i,2*i+1]);
#        if abs(Lambdai-1)<10^(10):
#            Lambdai = 1;
        
        Lambda[2*i] = Lambdai;
        Lambda[2*i+1] = Lambdai;
    
    T = M@T@M;
    Q = Q@M;
    
    # such that K = Q.'*T*Q
    Q = Q.T;
    
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
    if diag_error > ERROR_CUTOFF:
        print(" --> [takagi_decomp_] D diag error: " + str(diag_error))   
        
    unitary_error = check_zero(Up @ Up.conj().T - np.eye(len(Up)))
    if unitary_error > ERROR_CUTOFF:
        print(" --> [takagi_decomp_] U unitary error: " + str(unitary_error))
        
    decomp_error = check_zero(A - Up.T @ D @ Up)
    if decomp_error > ERROR_CUTOFF:
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
        
        
        
    