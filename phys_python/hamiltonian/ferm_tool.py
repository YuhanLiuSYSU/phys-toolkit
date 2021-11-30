import numpy as np
import cmath
from math import pi
import itertools


from eig.decomp import sort_ortho, simult_diag, fold_brillouin, move_brillouin
from toolkit.check import check_diag

Sy = np.array(([0, -1j], [1j, 0]), dtype=np.complex128)
I2 = np.array(([1, 0], [0, 1]), dtype=np.complex128)


class Ferm_hamiltonian:
    """
    Input is fermionic Hamiltonian matrix, or parameters for the matrix
    """
    
    def __init__(self, H = None, N = None, P = "+", 
                 bands = 1, w= None,v=None,u=None, is_herm = True):
        
        self.H = H
        if isinstance(H,np.ndarray):
            self.N = len(H)
        elif bool(N):
            self.N = N
            
        self.N_k = int(self.N/bands)
        self.P = P      # boundary condition. PBC or APBC
        self.k_tot = np.arange(self.N_k)*2*pi/self.N_k
        if P == "-":
            self.k_tot = self.k_tot + pi/self.N_k
            
        self.bands = bands
        self.is_herm = is_herm
        
        self.w = w
        self.v = v
        self.u = u
        
    def single_eig(self):
        [eigval, eigvec] = sort_ortho(self.H)

        return eigval, eigvec      
    
    def get_LR_ssh(self):
        s_eig = []
        P_eig = []
        self.R_plus = []
        self.R_minus = []
        self.L_plus = []
        self.L_minus = []
        r_eigvec = np.zeros((self.N, self.N),dtype=np.complex128)
        l_eigvec = np.zeros((self.N, self.N),dtype=np.complex128)
        
        w = self.w
        v = self.v
        u = self.u
        
        # Generate single body spectrum
        # corr = np.zeros((self.bands*max(sub_N), self.bands*max(sub_N)),dtype=np.complex128)
        for i,k in enumerate(self.k_tot):
            # Only include the positive part of the energy
            s0_eig = np.sqrt(abs(abs(w*np.exp(-1j*k)+v)**2-u**2))
        
            s_eig.append(-s0_eig)
            s_eig.append(s0_eig)
            
            P_eig.append(k*self.N_k/(2*pi))
            P_eig.append(k*self.N_k/(2*pi))
            
            
            # Note that we include one state at zero energy!!
            if abs(w*np.exp(-1j*k)+v) >= u:
                
                # After diagonalizing P, l is not necessary conjugate of r
                r_minus, l_minus, r_plus, l_plus = find_v_ssh(k, u, v, w)
                self.R_plus.append(r_plus)
                self.R_minus.append(r_minus)
                self.L_plus.append(l_plus)
                self.L_minus.append(l_minus)
                 
                prod = np.exp(1j*k*np.array([range(1,self.N_k+1)])).T
                
                vr_minus = (np.kron(prod, r_minus))[:,0]
                vr_plus = (np.kron(prod, r_plus))[:,0]
                                
                r_eigvec[:,2*i] = vr_minus/np.sqrt(self.N_k)
                r_eigvec[:,2*i+1] = vr_plus/np.sqrt(self.N_k)
                
                if self.is_herm == False:
                    vl_minus = (np.kron(prod, l_minus))[:,0]
                    vl_plus = (np.kron(prod, l_plus))[:,0]
                                    
                    l_eigvec[:,2*i] = vl_minus/np.sqrt(self.N_k)
                    l_eigvec[:,2*i+1] = vl_plus/np.sqrt(self.N_k)
                
                
        s_eig = np.array(s_eig)
        P_eig = np.array(P_eig)
        
        P_eig = move_brillouin(P_eig, self.N_k)
        
        if self.is_herm:
            print(" [get_LR_ssh] error for orthonormal: %f" % 
                  check_diag(r_eigvec.conj().T @ r_eigvec, 0))
            l_eigvec = r_eigvec
        else:
            print(" [get_LR_ssh] error for biorthonormal: %f" % 
                  check_diag(l_eigvec.conj().T @ r_eigvec, 0))

        
        return s_eig.real, r_eigvec, l_eigvec, P_eig
    
    
    def many_eig(self, cutoff = 20, is_fold = 0):
        
        N = self.N
        
        # Comment: For the following code, there exists unresolved degeneracy. 
        # The result for RE-MI is not accurate. lst[0] and lst[1] gives different result
        if isinstance(self.H,np.ndarray):
        #-----------------------------------------------------
            trans_op = self.trans_op(bands = self.bands)
            s_eig, r_eigvec, P_eig = simult_diag(self.H, trans_op, is_phase = 1,
                                                bands = self.bands, is_zero_sym=1)
        #-----------------------------------------------------
        else:
        # Comment: the result for RE-MI is accurate, same for lst[0], lst[1], 
        # lst[2], lst[3]
        #-----------------------------------------------------
            s_eig, r_eigvec, l_eigvec, P_eig = self.get_LR_ssh()
        #-----------------------------------------------------  

        
        
        gs_energy = -abs(s_eig).sum()/2
        gs_parity = 1-2*(int(N/2) % 2)
        
        # Sort single body eigenstates according to abs value
        # While keep track of the GS location.
        s_idx = np.argsort(abs(s_eig))
        s_eig = s_eig[s_idx]
        P_eig = P_eig[s_idx]
        r_eigvec = r_eigvec[:, s_idx]
        if 'l_eigvec' in locals():
            l_eigvec = l_eigvec[:, s_idx]
              
        gs_idx = np.where(s_eig<0)[0]       
        gs_list = np.zeros(N)
        gs_list[gs_idx.tolist()] = 1
              
        # Many body eigenstates
        n_cut = int(np.log2(cutoff))
        lst = np.array(list(itertools.product([0, 1], repeat=n_cut)))
            
        lst_parity = 1-lst*2
        lst_parity = lst_parity.prod(axis = 1)*gs_parity
        
        exs_eig = abs(s_eig)
        m_eig = exs_eig[0: n_cut] @ lst.T + gs_energy
               
        m_idx = np.argsort(m_eig)
        m_eig = m_eig[m_idx]
        lst_parity = lst_parity[m_idx]
        lst = lst[m_idx]
        lst = np.hstack((lst, np.zeros((2**n_cut, N - n_cut))))
        lst = (lst+gs_list) % 2
        ferm_nb = lst.sum(axis = 1) - int(N/2)
        P_meig = lst @ P_eig
        
        # Move to "first Brillouin zone"
        N_brillouin = N/self.bands
        P_meig = move_brillouin(P_meig, N_brillouin)
        
        # Fold Brillouin zone
        if is_fold == 1:
            P_meig = fold_brillouin(P_meig, N)
        
        combine = np.vstack((m_eig, lst_parity, ferm_nb, P_meig)).T
              
        self.s_eig = s_eig
        self.P_eig = P_eig
        self.r_eigvec = r_eigvec
        if 'l_eigvec' in locals():
            self.l_eigvec = l_eigvec
        else:
            self.l_eigvec = r_eigvec
        self.combine = combine
        self.lst = lst
        
        return s_eig, r_eigvec, combine, lst
    
    
    def get_gamma(self,m_fill):
        """
        m_fill is the filled bands, for example, lst[0]
        
        return the covariance matrix Gamma
        """
        r_eigvec = self.r_eigvec
        l_eigvec = self.l_eigvec
        
        N = len(m_fill)
        
        if len(m_fill.shape) == 1:
            m_fill = np.diag(m_fill)
        
        Corr = r_eigvec @ m_fill @ l_eigvec.conj().T

        Gamma = np.kron(Corr-Corr.transpose(),I2)\
            +np.kron(np.eye(N)-Corr-Corr.transpose(),Sy)
        
        return Corr, Gamma
    
    
    def trans_op(self, bands = 1):
        # TODO: think about APBC when bands == 2
        
        N = self.N
        trans_op = np.diag(np.tile(1, N-1), k = 1)
      
        if self.P == "+":
            trans_op[N-1,0] = 1
        else:
            trans_op[N-1,0] = -1
            
        if bands > 1:
            trans_op = np.linalg.matrix_power(trans_op,bands)
        
        return trans_op
    

    def select_state(self):
        combine = self.combine
        if self.P == "+":
            loc = np.where(combine[:,1]==-1)[0]
        else:
            loc = np.where(combine[:,1]==1)[0]
        combine = combine[loc]
        
        return combine
    

#---------------------------------------------------------------------------#


def find_v_ssh(k, u, v, w):
    vk = w*np.exp(-1j*k)+v
    # print(vk)
    avk = abs(vk)
    # print(avk)
    e_phi = cmath.sqrt(cmath.sqrt((u+avk)/(u-avk)))
    c_phi = (e_phi+1/e_phi)/2
    s_phi = (e_phi-1/e_phi)/(2*1j)
    # print(c_phi)
    # print(s_phi)
    # print()
    
    vr_minus = np.array([[-vk/avk*s_phi],[c_phi]])
    vl_minus = np.array([[-vk/avk*s_phi.conjugate()],[c_phi.conjugate()]])
    
    vr_plus = np.array([[vk/avk*c_phi],[s_phi]])
    vl_plus = np.array([[vk/avk*c_phi.conjugate()],[s_phi.conjugate()]])
    
    return vr_minus, vl_minus, vr_plus, vl_plus


def H_ssh(N, w, v, u, PBC = "+"):
    # Compare the energy spectrum
    
    H0 = np.diag(np.tile(np.array([1j*u,-1j*u]), N))
    H = np.diag(np.hstack((np.tile(np.array([v,w]), N-1),np.array([v]))),k=1)
    if PBC == "+":
        # start from here      
        H[2*N-1, 0] = w
    else:
        H[2*N-1, 0] = -w
        
    H = H + H.transpose()+H0
    
    return H


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

