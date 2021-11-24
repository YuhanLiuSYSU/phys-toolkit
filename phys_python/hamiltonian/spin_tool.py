import numpy as np
from scipy.sparse import csr_matrix
import scipy.linalg as alg
from math import pi
from scipy.sparse.linalg import eigs, eigsh
import matplotlib.pyplot as plt

from toolkit.check import check_diag
from eig.decomp import simult_diag, fold_brillouin


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
#           get_Sz
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
    """Class for the spin Hamiltonian"""
    
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
            print()
        

        
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
            hamiltonian = hamiltonian + self.ep2*get_total_Sz(self.N)
                
        self.hamiltonian = hamiltonian
       # self.hamiltonian = hamiltonian.astype(np.complex128)
        # This is very necessary
        
        self.h2 = csr_matrix((np.array(valn_array), (x_array, y_array)), 
                             shape=(2**N, 2**N),dtype=np.complex128)
        
        
    def get_P(self):
        """
        Generate translational operator P
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
        elif bands == 1:
            y_array_P = np.hstack((np.arange(0,2**N,2),
                                   np.arange(1,2**N,2)))
            
        val_array_P = np.repeat([1,PBC*1,PBC*1,1],2**(N-2))
        self.P = csr_matrix((val_array_P, (x_array_P, y_array_P)), shape=(2**N, 2**N))
        
        return self.P
      
        
    def sort_biortho(self,knum, eig_which='SR', PT='true'):
        
        eigval, eigvecs = eigs(self.hamiltonian, k=knum, which=eig_which)
        idx = eigval.argsort()[::1]   
        eigval = eigval[idx]
        eigvecs = eigvecs[:,idx]
        
        if PT!='true':
            eigval_L, eigvecs_L = eigs(self.hamiltonian.conj().T, k=knum, which=eig_which)
            idx = eigval_L.argsort()[::1]   
            eigval_L = eigval_L[idx]
            eigvecs_L = eigvecs_L[:,idx]
        
        
        eig_norm = np.diag(1/np.sqrt(np.diag(eigvecs.conj().T@eigvecs)))
        eigvecs = eigvecs@eig_norm
        
        labels=[-1]
        eigvecs_sort = eigvecs+np.zeros(eigvecs.shape,dtype=complex)
        for i in range(len(eigval)-1):
            if abs(eigval[i+1]-eigval[i])>0.0000001:
                labels.append(i)
        
        for i in range(len(labels)-1):
            if labels[i+1]-labels[i]>1:
                reg = range(labels[i]+1,labels[i+1]+1)
                regVR = eigvecs[:,reg] 
                
                if np.sum(abs(regVR.T@regVR-np.identity(len(reg))))>0.0000001:
                    
                    eig_unnorm = self.__Takagifac(regVR)
                    eig_fac = np.diag(1/np.sqrt(np.diag(eig_unnorm.T@eig_unnorm)))               
                    
                    eig_norm = eig_unnorm@eig_fac
                    overlap = eig_norm.T @ eig_norm
                    # tsave = eig_norm[:,:]
                    
                    subreg = []
                    for j in range(len(reg)-1):
                        # Sort again
                        if abs(overlap[j,j+1])>0.000001:
                            subreg.extend([j,j+1])
                    subreg = list(set(subreg))
                    if subreg!=[]:
                        eig_unnorm_2 = self.__Takagifac(eig_norm[:,subreg])
                        eig_fac_2 = np.diag(1/np.sqrt(np.diag(eig_unnorm_2.T@eig_unnorm_2)))   
                        eig_norm_22 = eig_unnorm_2@eig_fac_2
                        eig_norm[:,subreg] = eig_norm_22
                        
                        # test4 = test
                        # test4[:,subreg] = eig_norm_22
                        # test3 = test4.T @ test4
                        # plt.imshow(abs(test3), cmap = 'jet')
                        # plt.colorbar()
                                              
                    eigvecs_sort[:,reg] = eig_norm
                    
        eig_norm = np.diag(1/np.sqrt(np.diag(eigvecs_sort.T@eigvecs_sort)))
        eigvecs_sort = eigvecs_sort@eig_norm        

        print("error for orthonormal: %f" % 
          check_diag(eigvecs_sort.T @ eigvecs_sort))
        print("error for H: %f" % 
          check_diag(abs(eigvecs_sort.T@ self.hamiltonian @eigvecs_sort)))
        
        self.eigval = eigval + self.const_term
        self.R = eigvecs_sort
        self.L = eigvecs_sort.conj()
        
        return self.eigval, eigvecs_sort
    
    
    def __Takagifac(self,R):
    # Autonne-Takagi factorization
    # D = UAU^T where A is a complex symmetric matrix, U is a unitary. D is real non-negative matrix
    
        A = R.T @ R 
        _,V = alg.eig(A.conj().T @ A)
        _,W = alg.eig((V.T @ A @ V).real)
        U = W.T @ V.T
        Up = np.diag(np.exp(-1j*np.angle(np.diag(U @ A @ U.T))/2)) @ U    
        
        return R@Up.T

        
    
    
    def sort_P_prod(self,knum):
        
        # TODO: to be finished...
        
        self.hamiltonian = self.hamiltonian.astype(np.float64)
        
        E, V = eigsh(self.hamiltonian, k=knum, which='LM')
        test = V.conj().T@ self.hamiltonian @ V
        print("before sort: error for orthonormal: %f" % check_diag(V.conj().T @ V))
        print("before sort: error for H: %f" % check_diag(test))
    
    
    def sort_P(self,knum, is_quiet = 1):
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
        
        self.hamiltonian = self.hamiltonian.astype(np.float64)
        
        E, V = eigsh(self.hamiltonian, k=knum, which='SA')
        
        eig_M, V_sort = simult_diag(self.hamiltonian, E, V, 
                                    self.P, is_phase = 1, is_quiet = 0)
        
        self.eigval = E
        self.R = V_sort
        self.L = V_sort 
        # Even here, L and R are the same; we use the same notation so the code 
        # is compatible with the non-hermitian case.
        
        self.S = eig_M
        
        return E, V_sort, eig_M
    
    
    
    def sort_P_nonh(self,E,V):
        P = self.P
        R = V+np.zeros(V.shape,dtype=complex)
        L = R.conj()
        # Need to specify V is complex. Otherwise it will take real part of V[:,reg]=regV@Vtrans
        labels=[]
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
        S = np.angle(P_eig.diagonal())*self.N/(2*pi)
        print("error for orthonormal: %f" % 
          check_diag(L.conj().T @ R))
        print("error for H: %f" % 
          check_diag(L.conj().T @ self.hamiltonian @ R))
        print("error for P: %f" % 
          check_diag(P_eig))
                   
        
        self.R, self.L, self.S = R, L, S
        
        return R, L, S
    
#----------------------------------------------------------------------------#
    def find_Sz(self):
        
        eigval = self.eigval
        eig_P = self.L.conj().T @ self.P @ self.R
        S = np.angle(eig_P.diagonal())*self.N/(2*pi)
        eig_Sz = np.diag(self.L.conj().T @ self.S_z @ self.R)
        eig_Sz_round = np.around(eig_Sz)
               
        # May need to check manually...    
        posi_Sz_p = np.where(eig_Sz_round == 1)[0]
        eigval_p, S_p = eigval[posi_Sz_p], S[posi_Sz_p]
        posi_Sz_m = np.where(eig_Sz_round == -1)[0]
        eigval_m, S_m = eigval[posi_Sz_m], S[posi_Sz_m]
        
        self.Sz_plus = np.vstack((eigval_p,S_p)).T
        self.Sz_minus = np.vstack((eigval_m,S_m)).T
        self.combine = (np.vstack(((eigval),S,eig_Sz))).T
        self.combine_simp = (np.vstack(((eigval).real,np.around(S),eig_Sz_round))).T
        
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
    #    ctest =  2*vI.conj().T@ Ln_total[1] @ Ln_total[0] @ vI  #Ln is already normalized with a
    
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
    
    
    def fold_S(self):
        """
        Fold the "Brillouin zone" for antiferromagetic case. 

        Returns
        -------
        S : numpy.array

        """        
        S = fold_brillouin(self.S, self.N)
 
        self.S = S
        
        return S
        
    
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
def get_Sz(N, flag=0):
    """ Generate S_z matrix (this is actually the parity!) 
    S^z_1 \oprod S^z_2 \cdots S^z_N
    """
      
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



def check_totalSz(model, level):
    """ 
    Example input: 
        check_totalSz(XXZ,[1,2])
    """
    
    if isinstance(level, int): level = [level]
    
    vec = model.R[:, level]
    total_Sz = get_total_Sz(model.N)
    
    eig_Sz = vec.conj().T @ total_Sz @ vec
    if len(level)>1:
        eig_Sz = (alg.eig(eig_Sz))[0]
    
    return eig_Sz



def get_total_Sz(N, flag=0):
    """ 
    Generate S^z_1+S^z_2+\cdots + S^z_N    
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

#----------------------------------------------------------------------------#
def my_imagesc(matr):
   
        plt.imshow(abs(matr), cmap = 'jet')
        plt.colorbar()
        # plt.rcParams["figure.figsize"] = (10,10)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        plt.show()
        
   
