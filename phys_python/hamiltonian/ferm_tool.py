import numpy as np
import cmath
from math import pi
import itertools
from scipy.linalg import eig

from eig.decomp import sort_ortho, simult_diag, fold_brillouin, move_brillouin,sort_biortho
from toolkit.check import check_diag

Sx = np.array(([0, 1], [1, 0]), dtype=np.complex128)
Sy = np.array(([0, -1j], [1j, 0]), dtype=np.complex128)
Sz = np.array(([1, 0], [0, -1]), dtype=np.complex128)
I2 = np.array(([1, 0], [0, 1]), dtype=np.complex128)


class Ferm_hamiltonian:
    """
    Input is fermionic Hamiltonian matrix, or parameters for the matrix
    """
    
    def __init__(self, H = None, N = None, P = "+", 
                 bands = 1, w= None,v=None,u=None, is_herm = True,offset=0):
        # TODO: is_depr_bands should be merged!!
        
        self.H = H
        if isinstance(H,np.ndarray):
            self.N = len(H)
            print(" [Ferm_hamiltonian] using H...")
        elif bool(N):
            self.N = N
                    
        self.N_k = int(self.N/bands)

        
        self.P = P      # boundary condition. PBC or APBC
        self.k_tot = np.arange(self.N_k)*2*pi/self.N_k+offset
        
        # --- debug ---
        # self.k_tot = np.arange(1,self.N_k)*2*pi/self.N_k-pi
        # self.k_tot = np.arange(int(self.N_k/2))*2*pi/self.N_k - pi/2
        # -------------
        
        if P == "-":
            self.k_tot = self.k_tot + pi/self.N_k
            
        self.bands = bands
        self.is_herm = is_herm
        
        self.w = w
        self.v = v
        self.u = u
        
        
    def single_eig(self):
        [eigval, eigvec] = sort_ortho(self.H)
        self.r_eigvec = eigvec
        self.l_eigvec = eigvec

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
                # r_minus, l_minus, r_plus, l_plus = find_v_ssh_old(k, u, v, w, isdebug = 1)
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
        
        # Add a last line to lst, to study only the state close to exceptional pt
        excep_state = np.zeros((1, N))
        excep_state[0,1] = 1
        lst = np.vstack((lst, excep_state))
        
        # Move to "first Brillouin zone"
        N_brillouin = N/self.bands
        P_meig = move_brillouin(P_meig, N_brillouin)
        
        # Fold Brillouin zone
        if is_fold == 1:
            P_meig = fold_brillouin(P_meig, N/self.bands)
        
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
    
    
    def get_gamma_simp(self, model = 1, fill_half = 0, fill_double = 0):
        
        # Only for half-filled state
        
        Corr = np.zeros((self.N, self.N), dtype = np.complex)
        err = []
        E_tot = []
        
        for k in self.k_tot:
            Ek, W = find_v(k,self.u, self.v, self.w, 
                           model = model,fill_double = fill_double)
            E_tot.append(Ek)
            
            if model == 4 and fill_half == 1:
                fm_srf = 0-0.00001
                
                if Ek.real < fm_srf:
                    err.append(sum(sum(abs(Sx @ W.conj() @ Sx-W))))
                    
                    prod = np.exp(1j*k*np.array([range(1,self.N_k+1)])).T
                    Corr = Corr + np.kron(prod @ prod.T.conjugate(), W)
                    
            else:
                prod = np.exp(1j*k*np.array([range(1,self.N_k+1)])).T
                Corr = Corr + np.kron(prod @ prod.T.conjugate(), W)
            
                        
        Corr = Corr/self.N_k
        Gamma = np.kron(Corr-Corr.transpose(),I2)\
            +np.kron(np.eye(self.N)-Corr-Corr.transpose(),Sy)
        
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
    

    def select_state(self, spin_pbc = 1):
        combine = self.combine
        parity = spin_pbc
        if self.P == "+":
            loc = np.where(combine[:,1]==-parity)[0]
        else:
            loc = np.where(combine[:,1]==parity)[0]
        combine = combine[loc]
        
        return combine
    

#---------------------------------------------------------------------------#




def find_v(k,u,v,w, model = 1, fill_double = 0):
    
    if model == 1:
        # non-herm SSH
        
        # if abs(k-pi)<10**(-6) or abs(k+pi)<10**(-6):
        #     u = 0.5
        #     ep = 10**(-9)
        #     Ek = 0
        #     Hk = np.array([[1j/ep, 1/ep],
        #                    [1/ep, -1j/ep]])
        #     W = 1/2*(np.eye(2)-Hk)
            
        #     # W = np.zeros((2,2))
        #     print("ex pt")
            
            
        # else:
        vk = w*np.exp(-1j*k)+v
        avk = abs(vk)
        Ek = cmath.sqrt(avk**2-u**2)
        Hk = np.array([[1j*u, vk],
                       [vk.conjugate(), -1j*u]],
                      dtype=np.complex)
        W = 1/2*(np.eye(2)-Hk/Ek)
            
            
    elif model == -2:
        # Hermitian SSH model
        
        vk = w*np.exp(-1j*k)+v
        avk = abs(vk)
        Ek = cmath.sqrt(avk**2-u**2)
        Hk = np.array([[1j*u, vk],
                       [vk.conjugate(), -1j*u]],
                      dtype=np.complex)
        W = 1/2*(np.eye(2)-Hk/Ek)
        
        
    elif model == 2:
        
        gamma = w
        Ek = cmath.sqrt((v-w*np.cos(k))**2+(gamma*np.sin(k))**2-u**2)
        Hk = (v-w*np.cos(k))*Sx + (gamma*np.sin(k))*Sy + 1j*(u)*Sz
        
        W = 1/2*(np.eye(2)-Hk/Ek)
        
        
    elif model == 3:
        
        a0 = 1000
        b0 = 1
        B = 4
        
        f1 = (2*(1-np.cos(k)))**(B/2)
        f2 = (2*(1-np.cos(k)))**(B/2)+a0
        
        Hk = np.array([[0, f1],
                       [f2, 0]])
        
        Ek = abs(cmath.sqrt(f1*f2))

        W = 1/2*(np.eye(2)-Hk/Ek)
        
        
    elif model == 4:
        
        J, gamma = 1, 0.5
        ep = 10**(-7)
        Delta = gamma*(1+ep)
        # Ek = -J*np.cos(k)+np.sqrt(Delta**2-gamma**2)*np.sin(k)
        # PT = 'true'
        Ek = -J*np.cos(k)
        PT = 'false'
        
        if ep>0:
            Hk = -J*np.cos(k)*I2+(1j*gamma*Sz+Delta*Sx)*np.sin(k)
            # p = np.array([[1,0],[-1j,1]])
            # p_inv = np.array([[1,0],[1j,1]])
            # Hk = p_inv @ Hk @ p
            
            # Hk = np.array([[-J*np.cos(k)+1j*(gamma-Delta)*np.sin(k), Delta*np.sin(k)],
            #                 [2*(Delta-gamma)*np.sin(k),
            #                 -J*np.cos(k)-1j*(gamma-Delta)*np.sin(k)]])
            
            eigval, R, L = sort_biortho(Hk, PT = PT)
        
            i_st = 0
           
            Rv = (R[:,i_st]).reshape((2, 1))
            Lv = (L[:,i_st]).reshape((2, 1))
        
            W = Rv @ Lv.conj().T
        
            if fill_double == 1:
                Rv1 = (R[:,1]).reshape((2, 1))
                Lv1 = (L[:,1]).reshape((2, 1))
                
                # This is always identity
                W = Rv @ Lv.conj().T + Rv1 @ Lv1.conj().T
                
            # if abs(k-0)<10**(-8):
            #     print("change")
            #     W = np.array([[1,0],[0,0]])
                # W = np.array([[1,1],[1,1]])/np.sqrt(2)
            # if abs(k-pi)<10**(-8):
            #     print("change")
                # W = np.array([[0,0],[0,1]])
                # W = np.array([[1,1],[1,1]])/np.sqrt(2)
        
        else:
            amp = 40
            if abs(k-0)<10**(-8) or abs(k-pi)<10**(-8):
                print("change")
                Rv = np.array([[1],[0]])
                Lv = np.array([[1],[0]])
            else:
                Rv = np.array([[0],[1]]) * amp
                Lv = np.array([[0],[1]]) * amp
            
            W = Rv @ Lv.conj().T
            
    
    return Ek, W



def find_v_ssh(k, u, v, w, log_pn = 0):
    # This is for the non-Hermitian model, where u is actually iu.
    # For hermitian SSH, it only works for u=0 case. 
    
    vk = w*np.exp(-1j*k)+v
    avk = abs(vk)
    Ek = cmath.sqrt(avk**2-u**2)
    
    norm_p = cmath.sqrt(1+(Ek-1j*u)**2/avk**2)
    norm_m = cmath.sqrt(1+(Ek+1j*u)**2/avk**2)
    
    # The following two lines are for defective Hamiltonian (Jordan block)
    is_jordan = 0
    if abs(norm_p) == 0 : 
        print(" --- Jordan block...")
        is_jordan = 1
        norm_p = 40
        norm_m = norm_p
  
    vr_minus = np.array([[1],[(-Ek-1j*u)/vk]])/norm_m
    vr_plus = np.array([[1],[(Ek-1j*u)/vk]])/norm_p
    
    vl_minus = np.array([[1],[(-Ek+1j*u)/vk]])/norm_m.conjugate()
    vl_plus = np.array([[1],[(Ek+1j*u)/vk]])/norm_p.conjugate()
    
    if log_pn == 1 and is_jordan == 1:
        # Use the logarithmic partner
        
        v_norm = cmath.sqrt(1-2*1j)
        vr_minus = np.array([[1],[1-1j]])/v_norm
        vl_minus = np.array([[1],[1+1j]])/v_norm.conjugate()
        
        # vr_minus = np.array([[0],[1]])
        # vl_minus = np.array([[0],[1]])
        
        vr_plus = vr_minus
        vl_plus = vl_minus
        print("")
        

    return vr_minus, vl_minus, vr_plus, vl_plus


def find_v_ssh_old(k, u, v, w, isdebug = 0):
    # KEEP THIS FOR COMPARISON. DO NOT DELETE!!
    
    # This is for the non-Hermitian model, where u is actually iu.
    # For hermitian SSH, it only works for u=0 case. 
    #------------------------------------------------------------------------
    # For hermitian critical point at u=0, k=pi, the  matrix element of the Hamiltonian 
    # is all zero. To obtain the correct eigenvectors, we take the limit k->pi
    # The two eigenvectors are thus (1,i) and (1,-i)
    #------------------------------------------------------------------------
    # 
        
    if isdebug == 0:
        print(" --! Use the other file... ")
        vr_minus, vl_minus, vr_plus, vl_plus = None, None, None, None
    
    else:
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
               
        
    comp = N/(2*pi)*(1j*Ferm.u*(L[0].conj()*R[0]-L[1].conj()*R[1])
                     +Ferm.v*(L[0].conj()*R[1]+L[1].conj()*R[0])
                     +Ferm.w*(L[0].conj()*R[1]*np.exp(1j*(-k1_adj+n*pi/N))
                              +L[1].conj()*R[0]*np.exp(1j*(k1_adj-n*pi/N))))
    
    return comp


if __name__ == "__main__":
    # 0,0.5,1.8,1.3 # 0,0,1,1 # pi,0.5,1.8,1.3
    
    # test1 = find_v_ssh(pi/2,0.5,1.8,1.3 )
    test2 = find_v(pi,0.5,1.8,1.3, model = 4, fill_double = 0)
