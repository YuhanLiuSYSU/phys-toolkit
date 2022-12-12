import numpy as np
import matplotlib.pyplot as plt
from math import pi
import cmath
import scipy.linalg as alg


Sy = np.array(([0, -1j], [1j, 0]), dtype=np.complex128)
I2 = np.array(([1, 0], [0, 1]), dtype=np.complex128)


def get_spec(N = 12, is_save = 0):
    
    # N is the number of unitcell. The total number of atom is 2*N
    NAB = np.array(range(2, N+1-2, 2))

    [w, v, u] = [1.00000000, 2.00000000, 0.0]
    
    bands = 2
    
    # Get the left and right eigenvectors, and the correlation matrix
    Corr = get_LR_corr(u,v,w,bands,NAB,N)      
    Gamma = np.kron(Corr-Corr.transpose(),I2)+np.kron(np.eye(2*max(NAB))
                                                      -Corr-Corr.transpose(),Sy)
    
    LN = np.zeros(len(NAB))
    
    
    for i,i_ele in enumerate(range(len(NAB))):
        LAB = NAB[i];
        GammaR = Gamma[0:2*bands*LAB,0:2*bands*LAB]
              
        LN[i_ele] = get_LN_(GammaR, 4*int(LAB/2), 4*int(LAB/2))
        
    plt.plot(NAB, LN)
                  
    return NAB, LN


def get_LR_corr(u,v,w,bands,sub_N,N_k):
        # Temporary purpose. To be merged...

    k_tot =  np.arange(N_k)*2*pi/N_k
    
    # Generate single body spectrum
    corr = np.zeros((bands*max(sub_N), bands*max(sub_N)),dtype=np.complex128)
    for k in k_tot:
        if abs(w*np.exp(-1j*k)+v) >= u:
            r_minus, l_minus, r_plus, l_plus = find_v_ssh(k, u, v, w)

            W = l_minus @ r_minus.conj().T
            prod = np.exp(1j*k*np.array([range(1,max(sub_N)+1)]))
            corr += np.kron(prod.conj().T @ prod, W)/N_k
            
    return corr


def find_v_ssh(k, u, v, w):
    # This is for the non-Hermitian model, where u is actually iu.
    # For hermitian SSH, it only works for u=0 case. 
    #------------------------------------------------------------------------
    # For hermitian critical point at u=0, k=pi, the  matrix element of the Hamiltonian 
    # is all zero. To obtain the correct eigenvectors, we take the limit k->pi
    # The two eigenvectors are thus (1,i) and (1,-i)
    #------------------------------------------------------------------------
    # 
    
    vk = w*np.exp(-1j*k)+v
    avk = abs(vk)
    e_phi = cmath.sqrt(cmath.sqrt((u+avk)/(u-avk)))
    c_phi = (e_phi+1/e_phi)/2
    s_phi = (e_phi-1/e_phi)/(2*1j)

    
    vr_minus = np.array([[-vk/avk*s_phi],[c_phi]])
    vl_minus = np.array([[-vk/avk*s_phi.conjugate()],[c_phi.conjugate()]])
    
    vr_plus = np.array([[vk/avk*c_phi],[s_phi]])
    vl_plus = np.array([[vk/avk*c_phi.conjugate()],[s_phi.conjugate()]])
    
    return vr_minus, vl_minus, vr_plus, vl_plus





def get_LN_(GammaR,NA,NB):
    
    eig_gamma, _ = alg.eig(GammaR)
    
    G11 = GammaR[0:NA, 0:NA]
    G12 = GammaR[0:NA, NA:NA+NB]
    G21 = GammaR[NA:NA+NB, 0:NA]
    G22 = GammaR[NA:NA+NA, NA:NA+NB]
    
    Gp = np.vstack((np.hstack((-G11, 1j*G12)),
                    np.hstack((1j*G21, G22))))
    Gm = np.vstack((np.hstack((-G11, -1j*G12)),
                    np.hstack((-1j*G21, G22))))
    
    Id = np.eye(Gp.shape[0])
    Gc = Id - (Id-Gm) @ alg.inv(Id+Gp@Gm) @ (Id-Gp)
    
    
    Gc = 1/2*(Gc+Gc.conj().T)
        
    eig_gc = alg.eig(Gc)[0]
    eig_gamma = eig_gamma.real
    

    R1 = get_renyi_(0.5, eig_gc)
    R2 = get_renyi_(2, eig_gamma)
    

    LN = R1+R2/2

    return LN


def get_renyi_(renyi, gamma):
    " Renyi entropy "
           
    eta = (gamma+1)/2
    cutoff = 10**(-7)
    
    eta = eta[eta > cutoff]
    eta = eta[(1-eta) > cutoff]
       
    Rfac = eta*abs(eta)**(renyi-1)+(1-eta)*abs(1-eta)**(renyi-1)
    R = np.log(Rfac.prod())/2
    
    return R


if __name__ == "__main__":

    N = 12
    NAB, LN = get_spec(N = N, is_save = 1)
    print("Negativity: ", LN)

        


        
    


   