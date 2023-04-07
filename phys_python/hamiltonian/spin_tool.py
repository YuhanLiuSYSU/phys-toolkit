import numpy as np
from scipy.sparse import csr_matrix
import scipy.linalg as alg
from math import pi
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

from toolkit.check import check_diag
from eig.decomp import simult_diag, fold_brillouin, simult_diag_old, \
    sort_biortho, simult_diag_nonh, sort_ortho


Sx = np.array(([0, 1], [1, 0]), dtype=np.complex128)
Sy = np.array(([0, -1j], [1j, 0]), dtype=np.complex128)
Sz = np.array(([1, 0], [0, -1]), dtype=np.complex128)

#----------------------------------#
# class: File_access
#   method: get_back
#
# class: Spin_hamiltonian
#   method: get_hamiltonian
#           get_P
#           get_prod_Sz
#           sort_biortho
#           check_diag
#           sort_P
#           sort_P_nonh
#           find_Sz
#           get_c
#           get_c_nonunitary

#----------------------------------#
# Log: What's new:
    # the last element of couple_diag and couple_off is changed



class Spin_hamiltonian:
    """
    Class for the spin Hamiltonian
    """
    
    def __init__(self, N, couple_diag,couple_off,PBC, bands=1, const_term = 0,
                 ep1 = None, ep2 = None):
        """
        

        Parameters
        ----------
        N : int
            Number of sites of the spin chain.
        couple_diag : list
                
        
        couple_off : list
            example: couple_off = [[np.repeat([Jx],N),Sx,Sx,np.full(N,1)]]
            
            The first element specifies the hopping strength. It is a numpy array
            If there are only two elements, it acts on single site i.
            If there are three elements, it acts on neighboring sites S^x_i S^x_{i+1}
            The fourth element specifies the hopping range S^x_{i} S^x_{i+n_i}.
                When it is absent, we use the default value n_i = 1.
                The input can be a single integer, or a numpy array.
            
        PBC : int
            DESCRIPTION.
        bands : int, optional
            DESCRIPTION. The default is 1.
        const_term : float, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """
        
        self.N = N
        self.bands = bands
        self.PBC = PBC
        self.couple_diag = couple_diag
        self.couple_off = couple_off 
        self.const_term = const_term
        self.a = 1 # set a default value
        
        self.ep1 = ep1
        self.ep2 = ep2
        
        
        if self.couple_diag is not None:
            for couple in self.couple_off: 
                if len(couple) == 3:
                    couple.append(np.full(N,1))
                elif len(couple) == 4:
                    if isinstance(couple[3], int):
                        couple[3] = np.full(N, couple[3])
                    
            for couple in self.couple_diag:
                if len(couple) == 3:
                    couple.append(np.full(N,1))
                elif len(couple) == 4:
                    if isinstance(couple[3], int):
                        couple[3] = np.full(N, couple[3])
                           
            self.get_hamiltonian()
            
            if PBC == 1 or PBC == -1:
                self.get_P()
    
            # this generate .hamiltonian and .h2, .P
        
    
    def get_hamiltonian(self):
        # couple1 = [J,Sx,Sx], etc. couple2 = [J,Sz]
        # couple_off=[[-Lambda/2,Sx,Sx],[-1j*h/2,Sx]]
        # couple_diag=[[-1/2,Sz]]
        
        N = self.N
        couple_diag = self.couple_diag
        couple_off = self.couple_off
        const_term = self.const_term
        
        n = 2
        Jdn = (N/(2*pi))* np.array([np.exp(1j*n*(i+1/2)*2*pi/N) for i in range(0,N)])
        Jsn = (N/(2*pi))*np.array([np.exp(1j*n*(i)*2*pi/N) for i in range(0,N)])
    
    
        hamiltonian_diag = np.zeros((2**N), dtype=np.complex128)
        hn_diag = np.zeros((2**N), dtype=np.complex128)
        # Important to specify it is complex
        
        x_array = []
        y_array = []
        val_array = []
        valn_array = []
        
        
        for j in range(len(couple_off)):  
            couple = couple_off[j]
            for i in range(len(couple[0])):
                if couple[0][i]!=0:                  
                    if len(couple)>2:
                    
                        x_array_new, y_array_new, val_array_new = set_Hamiltonian_offdiag(
                            couple[0][i], N, couple[1], i, couple[2], (i+couple[3][i]) % N)
                        _,_, valn_array_new = set_Hamiltonian_offdiag(
                            couple[0][i]*Jdn[i], N,couple[1], i, couple[2], (i+couple[3][i]) % N)
                        
                    elif len(couple)==2:
                        x_array_new, y_array_new, val_array_new = set_Hamiltonian_offdiag(
                            couple[0][i], N, couple[1], i)
                        _,_, valn_array_new = set_Hamiltonian_offdiag(
                            couple[0][i]*Jsn[i], N,couple[1], i)
                        
                    x_array.extend(x_array_new)
                    y_array.extend(y_array_new)
                    val_array.extend(val_array_new)
                    valn_array.extend(valn_array_new)
    
    
        for j in range(len(couple_diag)):
            couple = couple_diag[j]
            for i in range(len(couple[0])):
                if couple[0][i]!=0:
                    if len(couple)>2:
                        hamiltonian_diag += set_Hamiltonian_diag(
                            couple[0][i],N,couple[1],i,couple[2],(i+couple[3][i]) % N)
                        hn_diag += set_Hamiltonian_diag(
                            couple[0][i]*Jdn[i],N,couple[1],i,couple[2],(i+couple[3][i]) % N)
                    elif len(couple)==2:
                        hamiltonian_diag += set_Hamiltonian_diag(couple[0][i],N,couple[1],i)
                        hn_diag += set_Hamiltonian_diag(couple[0][i]*Jsn[i],N,couple[1],i)
    
    
        x_array.extend(range(2**N))
        y_array.extend(range(2**N))
        val_array.extend(hamiltonian_diag)
        valn_array.extend(hn_diag)
    
        hamiltonian = csr_matrix((np.array(val_array), (x_array, y_array)), 
                                 shape=(2**N, 2**N),dtype=np.complex128)
        
        
        if bool(self.ep1): 
            hamiltonian = hamiltonian + self.ep1*self.get_P()
        
        if bool(self.ep2):
            hamiltonian = hamiltonian + self.ep2*get_sum_Sz(self.N)
                
        self.h = hamiltonian
       # self.h = hamiltonian.astype(np.complex128)
        # This is very necessary
        
        self.h2 = csr_matrix((np.array(valn_array), (x_array, y_array)), 
                             shape=(2**N, 2**N),dtype=np.complex128)
        
        
    def get_spec(self, knum):
        # For Ising OBC
        eigval, eigvec = sort_ortho(self.h, knum=knum)
        # eigval = eigval - min(eigval)
        # eigval = eigval*2/(eigval[3]-eigval[0])
        
        self.eigval = eigval
        self.R = eigvec
        self.L = eigvec
        
        return eigval, eigvec
        
        
    def get_P(self):
        """
        Generate translational operator P. Translate towards right. 
        The basis are: |uuuu>, |uuud>, |uudu>, |uudd>, ...
        
        -----------
        For APBC, the following construction works for the Ising model, where:
            |uuud> -> |uudu>,       |duud> -> -|uudd>
            |uudd> -> |uddu>,       |duuu> -> -|uuud>
        This is because the local symmetry generator is Sz, which we use to twist
        the boundary condition:
            |u>_{L+1} -> |u>_1
            |d>_{L+1} -> -|d>_1
        """
        
        N = self.N
        bands = self.bands
        PBC = self.PBC
        x_array_P = range(0,2**(N),1)
        if bands == 2:
            y_array_P = np.hstack((np.arange(0,2**N,4),
                               np.arange(1,2**N,4),
                               np.arange(2,2**N,4),
                               np.arange(3,2**N,4)))
            val_array_P = np.repeat([1,PBC*1,PBC*1,1],2**(N-2))
            
        elif bands == 1:
            y_array_P = np.hstack((np.arange(0,2**N,2),
                                   np.arange(1,2**N,2)))
            val_array_P = np.repeat([1,PBC*1],2**(N-1))
            # TODO: This works, but need to understand why.
            
        
        self.P = csr_matrix((val_array_P, (x_array_P, y_array_P)), shape=(2**N, 2**N))
        
        return self.P
      
        
    def sort_biortho_spin(self,knum, eig_which='SR', PT='true', is_sort = 1):
        
        eigval, R, L = sort_biortho(self.h, knum = knum, 
                                    eig_which = eig_which, PT = PT, is_sort=is_sort)
        
        # TODO: make the following code work...
        # eigval, R = simult_diag_nonh(self.h, 
        #                              [self.P, get_sum_Sz(self.N), get_prod_Sz(self.N)],
        #                              knum = knum)
        
        self.eigval = eigval + self.const_term
        self.R = R
        self.L = R.conj()
        
        return self.eigval, R
    

    def sort_P_old(self,knum, is_quiet = 1, is_quiet_debug = 1):
        """
        For Hermitian case. After finding eigensystems of hamiltonian, we silmultaneously
        diagonalize h and P(translational operator). 

        Parameters
        ----------
        knum : int
            DESCRIPTION.

        Returns
        -------
        E : TYPE
            DESCRIPTION.
        V_sort : TYPE
            DESCRIPTION.
        S : TYPE
            DESCRIPTION.

        """
        
        self.h = self.h.astype(np.float64)
        
        E, V = eigsh(self.h, k=knum, which='SA')
        
        eig_M, V_sort = simult_diag_old(self.h, E, V, 
                                    [self.P], 
                                    is_phase = 1, is_quiet = is_quiet, 
                                    is_quiet_debug = is_quiet_debug)
        # or [self.P, get_sum_Sz(self.N)]
        
        
        self.eigval = E
        self.R = V_sort
        self.L = V_sort 
        # Even here, L and R are the same; we use the same notation so the code 
        # is compatible with the non-hermitian case.
        
        self.S = eig_M
        
        return E, V_sort, eig_M
    
    
    def sort_P(self, knum, is_sum_Sz = 1, is_prod_Sz = 0):
        
        if is_sum_Sz == 1:
            M = [self.P, get_sum_Sz(self.N)]
        elif is_prod_Sz == 1:
            M = [self.P, self.get_prod_Sz_()]
        else:
            M = self.P
            
        eigval, eigvec, eig_M = simult_diag(
            self.h, M, knum = knum, is_phase = 1,
            is_show = 1)
    
    
        self.eigval = eigval
        self.R = eigvec
        self.L = eigvec
          
        if is_sum_Sz == 1:
            self.S = eig_M[0]
            self.total_Sz = eig_M[1]
            self.hp_pair = np.vstack((eigval, eig_M[0].real, eig_M[1].real)).T
            
        elif is_prod_Sz == 1:
            self.S = eig_M[0]
            self.prod_Sz = eig_M[1]
            self.hp_pair = np.vstack((eigval, eig_M[0].real, eig_M[1].real)).T
            
        else:
            self.S = eig_M
            self.hp_pair = np.vstack((eigval, eig_M.real)).T
        
        
    
        return eigval, eigvec, eig_M
    
    
    def sort_P_nonh(self,E,V, M = None, is_phase = 1):
        
        if M == None:
            P = self.P    # Translational symmetry
        else:
            P = M           # User input symmetry matrix M
        
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
        if is_phase == 1: # for translational symmetry
            S = np.angle(P_eig.diagonal())*self.N/(2*pi)
        else:
            S = P_eig.diagonal()
            
        print("error for orthonormal: %f" % 
          check_diag(L.conj().T @ R, is_show = 1))
        print("error for H: %f" % 
          check_diag(L.conj().T @ self.h @ R, is_show = 1))
        print("error for P: %f" % 
          check_diag(P_eig, is_show = 1))
                   
        
        self.R, self.L, self.S = R, L, S
        
        return R, L, S
    
#----------------------------------------------------------------------------#
    
    def get_prod_Sz_(self, flag=0):
        """ Generate S_z matrix (this is actually the parity!) 
        S^z_1 \oprod S^z_2 \cdots S^z_N
        """
          
        S_z = get_prod_Sz(self.N, flag = flag)
      
        self.prod_S_z = S_z
        
        return S_z
    
    
    def get_sum_Sz_(self, flag = 0):
        """
        Generate sum of S^z_i
        """
        
        S_z_sum = get_sum_Sz(self.N, flag = flag)
        
        return S_z_sum
        
        

    def find_Sz(self):
        
        self.get_prod_Sz_()
        
        eigval = self.eigval
        # eig_P = self.L.conj().T @ self.P @ self.R
        # S = np.angle(eig_P.diagonal())*self.N/(2*pi)
        S = self.S
        
        eig_Sz = np.diag(self.L.conj().T @ self.prod_S_z @ self.R)
        eig_Sz_round = np.around(eig_Sz)
               
        # May need to check manually...    
        posi_Sz_p = np.where(eig_Sz_round == 1)[0]
        eigval_p, S_p = eigval[posi_Sz_p], S[posi_Sz_p]
        posi_Sz_m = np.where(eig_Sz_round == -1)[0]
        eigval_m, S_m = eigval[posi_Sz_m], S[posi_Sz_m]
        
        self.Sz_plus = np.vstack((eigval_p,S_p)).T
        self.Sz_minus = np.vstack((eigval_m,S_m)).T
        
        self.combine = (np.vstack((eigval,S,eig_Sz))).T
        self.combine_simp = (np.vstack((eigval.real,np.around(S).real,
                                        eig_Sz_round.real))).T
        
        return self.Sz_plus, self.Sz_minus, eig_Sz, eig_Sz_round, S

    
    def get_c(self):
        eigvecs = self.R
        eigval = self.eigval
        
        vI = eigvecs[:,0]
        overlap = abs(eigvecs.conj().T @ self.h2 @ vI)
        position_max = np.argmax(overlap)
        a = 4*pi/(self.N*(eigval[position_max]-eigval[0]))
        c = 2*abs(eigvecs[:,position_max].conj().T @ self.h2 @ vI *a)**2
        Delta = eigval*a*self.N/(2*pi)
        combine = (np.vstack((Delta-min(Delta),self.S))).T
        
        self.c = c
        self.a = a
        self.combine = combine
        
        return c, a, Delta, combine

    # Use Virasoro generator to compute c... Result is the same.
    #    Use generate.get_Virasoro
    #    ctest =  2*vI.conj().T@ Ln_total[1] @ Ln_total[0] @ vI  
    #    Here Ln is already normalized with a
    
    def get_c_nonunitary(self):
        
        R, L, E = self.R, self.L, self.eigval
        
        vphi, vI = R[:,0], R[:,1]
        overlap = L.conj().T @ self.h2 @ vphi
        position_max = np.argmax(abs(overlap))
        a = 4*pi/(self.N*(E[position_max]-E[0]))
        
        n_total = [-2]
        h_m2 = get_Hn(self.N,self.couple_off,self.couple_diag,n_total)
        c = 2*(L[:,0]).conj().T@ h_m2[0] @ self.h2 @ vphi*a**2-8*(-1/5) 
        
        Delta = E*a*self.N/(2*pi)
        combine = (np.vstack(((Delta-Delta[1]),self.S))).T
        combine_simp = (np.vstack(((Delta-Delta[1]).real,np.around(self.S)))).T
        
        self.c = c
        self.a = a
        self.combine, self.combine_simp = combine, combine_simp
        
        return c, Delta, combine
    
    
    def fold_S(self, usr_N = None):
        """
        Fold the "Brillouin zone" for antiferromagetic case. 

        Returns
        -------
        S : numpy.array

        """        
        if bool(usr_N):
            S = fold_brillouin(self.S, usr_N)
        else:
            S = fold_brillouin(self.S, self.N)
 
        self.S = S
        
        return S
        
    
    def PT_eig(self, level=0):
        "PT symmetry for non-hermitian YL model"
    
        pt = self.get_prod_Sz()   
        r_ev = self.R[:,level]
        l_ev = self.L[:,level]
        pt_eig = l_ev.conj().T @ pt @ r_ev.conj()
        
        return pt_eig
    
    
    def get_first_Sx(self):
        "S_x \oprod 1 \oprod 1 ..."
        
        N = self.N
        x_array_Sz = range(0,2**(N),1)
        y_array_Sz = list(range(int(2**N/2), 2**N,1)) \
            + list(range(0,int(2**N/2),1))
        val_array_Sz = np.ones(2**N)
        
        first_sx = csr_matrix((val_array_Sz, (x_array_Sz, y_array_Sz)), 
                              shape=(2**N, 2**N))
    
        return first_sx
#----------------------------------------------------------------------------#








#----------------------------------------------------------------------------#
def set_Hamiltonian_offdiag(J, N, O1, index1, O2=-1, index2=-1):
    x_array = [0]
    y_array = [0]
    val1_array = [J]
    for i in range(N-1, -1, -1):
        x_array_new = [None]*(len(x_array)*2)
        y_array_new = [None]*(len(y_array)*2)
        val1_array_new = [None]*(len(val1_array)*2)
        if(i==index1):
            x_array_new[::2] = [2*x for x in x_array]
            y_array_new[::2] = [2*y+1 for y in y_array]
            val1_array_new[::2] = [val*O1[0][1] for val in val1_array]
            x_array_new[1::2] = [2*x+1 for x in x_array]
            y_array_new[1::2] = [2*y for y in y_array]
            val1_array_new[1::2] = [val*O1[1][0] for val in val1_array]
        elif(i==index2):
            x_array_new[::2] = [2*x for x in x_array]
            y_array_new[::2] = [2*y+1 for y in y_array]
            val1_array_new[::2] = [val*O2[0][1] for val in val1_array]
            x_array_new[1::2] = [2*x+1 for x in x_array]
            y_array_new[1::2] = [2*y for y in y_array]
            val1_array_new[1::2] = [val*O2[1][0] for val in val1_array]
        else:
            x_array_new[::2] = [2*x for x in x_array]
            y_array_new[::2] = [2*y for y in y_array]
            val1_array_new[::2] = val1_array
            x_array_new[1::2] = [2*x+1 for x in x_array]
            y_array_new[1::2] = [2*y+1 for y in y_array]
            val1_array_new[1::2] = val1_array

        x_array = x_array_new
        y_array = y_array_new
        val1_array = val1_array_new

    return x_array, y_array, np.array(val1_array) 

def set_Hamiltonian_diag(J, N, O1, index1,  O2=-1, index2=-1):
  # For example, when input index1=0, index2=1, and O=S_z, we will get
  # diag(I\otimes I\otimes \cdots S_z\cdots S_z)     
  
  # BUG: IT IS BETTER TO FIX THIS TO OUR CONVENTION. 
  # THIS IS THE REASON WHY THE OVERLAP IS TO T RATHER THAN TBAR...

    val_array = [J]
    for i in range(N-1, -1, -1):
        val_array_new = [None]*(len(val_array)*2)
        if(i==index1):
            val_array_new[::2] = [val*O1[0][0] for val in val_array]
            val_array_new[1::2] = [val*O1[1][1] for val in val_array]
        elif(i==index2):
            val_array_new[::2] = [val*O2[0][0] for val in val_array]
            val_array_new[1::2] = [val*O2[1][1] for val in val_array]
        else:
            val_array_new[::2] = val_array
            val_array_new[1::2] = val_array
        val_array = val_array_new
    
    return np.array(val_array)
#---------------------------------------------------------------------------#



def get_Hn(N,couple_off,couple_diag,n_total):
 # BUG: THE RESULT FOR H_2 IS ACTUALLY H_{-2}. 
 
    n_len = len(n_total)
    Jdn = (N/(2*pi))*np.array([[np.exp(1j*n*(i+1/2)*2*pi/N) for i in range(0,N)] for n in n_total])
    Jsn = (N/(2*pi))*np.array([[np.exp(1j*n*(i)*2*pi/N) for i in range(0,N)] for n in n_total])
        
    x_array = []
    y_array = []
    valn_array = [[] for i in range(n_len)]
    hn_diag = [np.zeros((2**N), dtype=np.complex128) for i in range(n_len)]
    
    for i in range(N):
         for ii in range(len(couple_off)):   
            couple = couple_off[ii]
            if couple[0][i]!=0:  
                if len(couple)>2:
                    for j in range(n_len):
                        x_array_new, y_array_new, val_array_new = set_Hamiltonian_offdiag(
                            couple[0][i]*Jdn[j,i], N, couple[1], i, couple[2],(i+couple[3][i])%N)
                        valn_array[j].extend(val_array_new)
                elif len(couple)==2:
                    for j in range(n_len):
                        x_array_new, y_array_new, val_array_new = set_Hamiltonian_offdiag(
                            couple[0][i]*Jsn[j,i], N, couple[1], i)
                        valn_array[j].extend(val_array_new)
                       
                x_array.extend(x_array_new)
                y_array.extend(y_array_new)
        
         for ii in range(len(couple_diag)):
             couple = couple_diag[ii]
             if couple[0][i]!=0:  
                 if len(couple)>2 and couple[0][i]!=0:
                     for j in range(n_len):
                         hn_diag[j] += set_Hamiltonian_diag(
                             couple[0][i]*Jdn[j,i],N,couple[1],i,couple[2],(i+couple[3][i])%N)
                 elif len(couple)==2 and couple[0][i]!=0:
                     for j in range(n_len):
                         hn_diag[j] += set_Hamiltonian_diag(
                             couple[0][i]*Jsn[j,i],N,couple[1],i)
        

    x_array.extend(range(2**N))
    y_array.extend(range(2**N))

    hn_total=[]
    for j in range(n_len):
        valn_array[j].extend(hn_diag[j])
        hn = csr_matrix((valn_array[j], (x_array, y_array)), 
                        shape=(2**N, 2**N),dtype=np.complex128)
        hn_total.append(hn)
        
    return hn_total


def get_Virasoro(hamiltonian,N, couple_off,couple_diag, n_total,a):
    # Using the first method in Appendix A
    h_0 = N/(2*pi)*hamiltonian
    hn_total = get_Hn(N,couple_off,couple_diag,n_total)
    hn_minus_total = get_Hn(N,couple_off,couple_diag,(-1)*np.array(n_total))
    
    Ln_total = []
    Lnb_total = []
    for j in range(len(n_total)):
        Ln_total.append(1/2*(a*hn_total[j] + a**2/n_total[j]*(hn_total[j]@h_0-h_0@hn_total[j])))
        Lnb_total.append(1/2*(a*hn_minus_total[j] + a**2/n_total[j]*(hn_minus_total[j]@h_0-h_0@hn_minus_total[j])))
             
    return Ln_total, Lnb_total

#----------------------------------------------------------------------------#


def get_trans(N, PBC=1):
    """
    Obtain the translational operator
    """
    
    anxl = Spin_hamiltonian(N, None, None, PBC)
    trans = anxl.get_P()
    
    return trans


def get_prod_Sz(N, flag=0):

    x_array_Sz = range(0,2**(N),1)
    y_array_Sz = x_array_Sz
    val_array_Sz = np.array([1,-1])
    for i in range(N-1): 
        val_array_Sz = np.hstack((val_array_Sz,-1*val_array_Sz))
        
    if flag==0:    
        S_z = csr_matrix((val_array_Sz, (x_array_Sz, y_array_Sz)), shape=(2**N, 2**N))
    else:
        S_z = val_array_Sz

    return S_z


def check_sum_Sz(model, level):
    """ 
    Example input: 
        check_sum_Sz(XXZ,[1,2])
    """
    
    if isinstance(level, int): level = [level]
    
    vec = model.R[:, level]
    total_Sz = get_sum_Sz(model.N)
    
    eig_Sz = vec.conj().T @ total_Sz @ vec
    if len(level)>1:
        eig_Sz = (alg.eig(eig_Sz))[0]
    
    return eig_Sz



def get_sum_Sz(N, flag=0):
    """ 
    Generate S^z_1+S^z_2+\cdots + S^z_N. This is "Q" in compactified boson model   
    """
      
    x_array_Sz = range(0,2**(N),1)
    y_array_Sz = x_array_Sz
    val_array_Sz = np.array([1,-1])
    for i in range(N-1): 
        val_array_Sz = np.repeat(val_array_Sz, 2)
        add_new = np.tile(np.array([1,-1]), 2**(i+1))
        
        val_array_Sz = val_array_Sz + add_new
        
    if flag==0:    
        S_z = csr_matrix((val_array_Sz, (x_array_Sz, y_array_Sz)), shape=(2**N, 2**N))
    else:
        S_z = val_array_Sz
  
    # convention: S_z = sigma_z/2
    S_z = S_z/2  
  
    return S_z



    

def extract_c_scaling(E0, E1, L, h, state = 0):
    
    # For Ising model, the finite size scaling is c\propto 1/L^2
    
    N = len(E0)
    c = np.zeros(N-2, dtype = np.complex128)
    for i in range(1,N-1):
        m = E1[i] - E0[i]
        
        # xi is almost a constant for different size
        xi = L[i]*m/(2*pi)*1/h
        
        if state == 0:
            c[i-1] = -6/(pi*xi)*(E0[i+1]+E0[i-1]-2*E0[i])/(1/L[i+1]+1/L[i-1]-2/L[i])
        else:
            c[i-1] = -6/(pi*xi)*(E1[i+1]+E1[i-1]-2*E1[i])/(1/L[i+1]+1/L[i-1]-2/L[i])
        
    return c

#----------------------------------------------------------------------------#
def my_imagesc(matr):
   
        plt.imshow(abs(matr), cmap = 'jet')
        plt.colorbar()
        # plt.rcParams["figure.figsize"] = (10,10)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        plt.show()
        
   
