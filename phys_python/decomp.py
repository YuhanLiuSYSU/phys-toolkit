# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 12:48:23 2021

@author: Yuhan Liu
"""
import numpy as np
import scipy.linalg as alg

from phys_python.common import check_zero, check_symmetric


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



def takagifac_decomp_(A):
    """
    Autonne-Takagi factorization for complex symmetric matrix. 
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
        print(" --! [takagifac_decomp_] Input is not symmetric matrix!")
        return 0, 0

    _, V = alg.eig(A.conj().T @ A)
    _, W = alg.eig((V.T @ A @ V).real)
    U = W.T @ V.T
    Up = np.diag(np.exp(-1j*np.angle(np.diag(U @ A @ U.T))/2)) @ U   
    Up = Up.conj()
    
    D = Up.conj() @ A @ Up.conj().T
    
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
        [D, U] = takagifac_decomp_(K)
        
        print(check_zero(K - U.T @ D @ U))
        
        
    