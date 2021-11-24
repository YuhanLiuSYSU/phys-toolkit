import numpy as np
import cmath
from math import pi
import itertools

from eig.decomp import sort_ortho, simult_diag, fold_brillouin

Sy = np.array(([0, -1j], [1j, 0]), dtype=np.complex128)
I2 = np.array(([1, 0], [0, 1]), dtype=np.complex128)


class Ferm_Hamiltonian:
    """
    Input is fermionic Hamiltonian matrix
    """
    
    def __init__(self, H):
        self.H = H
        self.N = len(H)
        
        
    def single_eig(self):
        [eigval, eigvec] = sort_ortho(self.H)

        return eigval, eigvec       
    
    
    def many_eig(self, cutoff = 20, P = "+", is_fold = 0):
        
        N = self.N
        
        s_eig, eigvec = self.single_eig()
        trans_op = self.trans_op(P = P)
        trans_eig, eigvec = simult_diag(self.H, s_eig, eigvec, 
                                        trans_op, is_phase = 1)    
        
        gs_energy = s_eig[0:int(N/2)].sum()
        gs_parity = 1-2*(int(N/2) % 2)
        
        s_idx = np.argsort(abs(s_eig))
        gs_idx = np.where(s_idx<int(N/2))[0]
        gs_list = np.zeros(N)
        gs_list[gs_idx.tolist()] = 1
        
        exs_eig = abs(s_eig[s_idx])
        eigvec = eigvec[:, s_idx]
        trans_eig = trans_eig[s_idx]
        
        n_cut = int(np.log2(cutoff))
        lst = np.array(list(itertools.product([0, 1], repeat=n_cut)))
            
        lst_parity = 1-lst*2
        lst_parity = lst_parity.prod(axis = 1)*gs_parity
        
        m_eig = exs_eig[0: n_cut] @ lst.T + gs_energy
               
        m_idx = np.argsort(m_eig)
        m_eig = m_eig[m_idx]
        lst_parity = lst_parity[m_idx]
        lst = lst[m_idx]
        lst = np.hstack((lst, np.zeros((2**n_cut, N - n_cut))))
        lst = (lst+gs_list) % 2
        ferm_nb = lst.sum(axis = 1) - int(N/2)
        trans_meig = lst @ trans_eig
        
        left = -N/2 - 10**(-4)
        right = N/2 + 10**(-4)
    
        loc_l_out = np.where(trans_meig<left)[0]
        trans_meig[loc_l_out] = trans_meig[loc_l_out] + N
        loc_r_out = np.where(trans_meig>right)[0]
        trans_meig[loc_r_out] = trans_meig[loc_r_out] - N
        
        if is_fold == 1:
            trans_meig = fold_brillouin(trans_meig, N)
        
        combine = np.vstack((m_eig, lst_parity, ferm_nb, trans_meig)).T
        
        
        self.s_eig = s_eig[s_idx]
        self.eigvec = eigvec
        self.combine = combine
        
        return s_eig[s_idx], combine, lst, eigvec
    

    
    
    def trans_op(self, P = "+"):
        
        N = self.N
        trans_op = np.diag(np.tile(1, N-1), k = 1)
        
        if P == "+":
            trans_op[N-1,0] = 1
        else:
            trans_op[N-1,0] = -1
        
        return trans_op
    
    
    
        
    
        
class FermHamiltonian:
    """Class for the fermionic Hamiltonian"""
    
    def __init__(self, N, u, v,w, offset,bands):
        self.N = N
        self.u = u
        self.v= v
        self.w = w
        self.offset = offset
        self.bands = bands
        
#        if PBC==1:
#            shift = 0
#            self.select = 4
#        elif PBC==-1:
#            shift = pi/N
#            self.select = 5
        
        self.k_tot = np.arange(N)*2*pi/N + self.offset
        
    def get_LR(self,sub_N):
        self.eig = []
        self.R_plus = []
        self.R_minus = []
        self.L_plus = []
        self.L_minus = []
        u,v,w = self.u, self.v, self.w
        
        # Generate single body spectrum
        corr = np.zeros((self.bands*max(sub_N), self.bands*max(sub_N)),dtype=np.complex128)
        for k in self.k_tot:
            # Only include the positive part of the energy
            self.eig.append(np.sqrt(abs(abs(w*np.exp(-1j*k)+v)**2-u**2)))
                          
            if abs(w*np.exp(-1j*k)+v) >= u:
                r_minus, l_minus, r_plus, l_plus = find_v_ssh(k, u, v, w)
                self.R_plus.append(r_plus)
                self.R_minus.append(r_minus)
                self.L_plus.append(l_plus)
                self.L_minus.append(l_minus)
                
                W = l_minus @ r_minus.conj().T
                prod = np.exp(1j*k*np.array([range(1,max(sub_N)+1)]))
                corr += np.kron(prod.conj().T @ prod, W)/self.N
                
        # plt.figure(0)
        # plt.plot(k_tot, eig,color='blue')
        # plt.xlabel('$k$',fontsize=20)
        # plt.ylabel('$E$',fontsize=20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.grid(True)
                
        return corr
    
    def many_body(self, cut_off_save):
        # How to optimize the code?
        
        N = self.N
        eig_sub = np.hstack((self.eig[0:N],self.eig[0:N]))
        k_disp = np.hstack((self.k_tot-pi,-(self.k_tot-pi)))
        #    k_disp = np.hstack((k_tot,-(k_tot)))
        m_eig = np.zeros(2**(len(eig_sub)))
        k_eig = np.zeros(2**(len(eig_sub)))
        m_eig[0:2], k_eig[0:2] = [eig_sub[0],0], [k_disp[0],0]
        
        
        for i in range(len(eig_sub)-1):
            m_eig[2**(i+1):2**(i+2)] = m_eig[0:2**(i+1)]+eig_sub[i+1]
            k_eig[2**(i+1):2**(i+2)] = k_eig[0:2**(i+1)]+k_disp[i+1]
            
            
        idx = m_eig.argsort()[::1]   
        m_eig = m_eig[idx]
        a = m_eig[self.select]-m_eig[0]
        m_eig = m_eig/a
        k_eig = (k_eig[idx])*N/(2*pi)
        
        k_eig_mod = (k_eig % (2*N)) - (2*N)*((k_eig % (2*N)) // N)
        self.combine = (np.vstack((m_eig[0:cut_off_save],k_eig[0:cut_off_save]))).T
        self.a = a
               
        return m_eig[0:cut_off_save], k_eig[0:cut_off_save]
        
    


#---------------------------------------------------------------------------#
def get_gamma(eigvec, m_fill):
    """
    m_fill is the filled bands
    
    return the covariance matrix Gamma
    """
    
    N = len(m_fill)
    Corr = eigvec @ m_fill @ eigvec.conj().T

    Gamma = np.kron(Corr-Corr.transpose(),I2)\
        +np.kron(np.eye(N)-Corr-Corr.transpose(),Sy)
    
    return Gamma


def find_v_ssh(k, u, v, w):
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



def find_s(combine, val):
    Delta = combine[:,0]
    S = combine[:,1]
    ep = 0.0000001
    
    position = [i for i in range(len(Delta)) if (S[i]>val-ep and S[i]<val+ep)]
    
    return Delta[position]


def hn_element(Ferm, k1,k2,s1,s2,n):
    # s1=1 means taking plus; s1=-1 means taking minus
    N = Ferm.N
    if n!= (k1-k2)*N/(2*pi):
        print("--- Check the input!")
    
    k_posi_1 = np.where(abs(Ferm.k_tot-pi-k1)<0.00001)[0][0]
    k_posi_2 = np.where(abs(Ferm.k_tot-pi-k2)<0.00001)[0][0]
    k1_adj = k1 + pi
    
    if s1==1:
        L = Ferm.L_plus[k_posi_1]
    else:
        L = Ferm.L_minus[k_posi_1]    
        
    if s2==1:
        R = Ferm.R_plus[k_posi_2]
    else:
        R = Ferm.R_minus[k_posi_2]
               
        
    comp = N/(2*pi)*(1j*Ferm.u*(L[0].conj()*R[0]-L[1].conj()*R[1])+Ferm.v*(L[0].conj()*R[1]+L[1].conj()*R[0])+Ferm.w*(L[0].conj()*R[1]*np.exp(1j*(-k1_adj+n*pi/N))+L[1].conj()*R[0]*np.exp(1j*(k1_adj-n*pi/N))))
    
    return comp

