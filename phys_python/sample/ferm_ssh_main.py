import numpy as np
import matplotlib.pyplot as plt
import time 
from math import pi
import scipy.io

from toolkit.file_io import File_access
from toolkit.plot_style import plot_style
import entangle.ent_ferm as ent_ferm
from entangle.ent_generate import GenerateEnt
import hamiltonian.ferm_tool as ferm_tool



Sy = np.array(([0, -1j], [1j, 0]), dtype=np.complex128)
I2 = np.array(([1, 0], [0, 1]), dtype=np.complex128)

# TODO: check tr(Gamma_c)=0

def get_spec(N = 12, is_save = 0):
    start_time = time.time()
    Dir = File_access()
    
    # N is the number of unitcell. The total number of atom is 2*N
    NAB = np.array(range(2, N+1-2, 2))
    # NAB = np.array(range(2, N+1-2, 1))
        
    #-------------------------------------------------------------------------#
    # CHOOSE FROM MODELS:
    model = 3
    #-------------------------------------------------------------------------#
    # First critical point
    if model == 1:
        [w, v, u, model_name, is_non_herm, PT, offset, renyi] = [
                1.30000000, 1.8+10**(-8), 0.5, 'non-herm SSH', 1, 'true', 0, 2]
    
    # [w, v, u, model_name, is_non_herm, PT, offset, renyi] = [
    #         1.3, 1.8, 0.5, 'non-herm SSH', 1, 'true', 10**(-8), 2]

    
    # Second critial point
    elif model == 2:
        [w, v, u, model_name, is_non_herm, PT, offset, renyi] = [
                1.8000000, 1.3-10**(-8), 0.5, 'non-herm SSH', 1, 'true', 0, 2]
    
    
    # [w, v, u, model_name, is_non_herm, PT, offset, renyi] = [
    #         1.8000000, 1.3000000, 0.5, 'non-herm SSH', 1, 'false', 10**(-7),2]
    
    # Hermitian SSH critical point
    elif model == 3:
        [w, v, u, model_name, is_non_herm, PT, offset, renyi] = [
                  1.00000000, 1.00000000, 0.0, 'herm SSH', 0, 'true', 0, 1/2]
    
    # [w, v, u, model_name, is_non_herm, PT, offset] = [
    #         1.00000000, 1.00000000, 0.0, 'herm SSH', 0, 'true', 10**(-7)]
    #-------------------------------------------------------------------------#
    
    ent_type = [1,2,3,4]
    
    bands = 2
    
    Ferm = ferm_tool.Ferm_hamiltonian(N=bands*N, u=u, v=v, w=w, offset=offset,
                                      bands = bands)    
    
    result_ent = GenerateEnt(ent_type, NAB, N, renyi = renyi)

    # Get the left and right eigenvectors, and the correlation matrix
    Corr = Ferm.get_LR_corr(NAB)      
    Gamma = np.kron(Corr-Corr.transpose(),I2)+np.kron(np.eye(2*max(NAB))
                                                      -Corr-Corr.transpose(),Sy)
    
    for i in range(len(NAB)):
        LAB = NAB[i];
        GammaR = Gamma[0:2*bands*LAB,0:2*bands*LAB]
        corr_sub = Corr[0:bands*LAB,0:bands*LAB];
              
        ent = ent_ferm.GetEntFerm(GammaR = GammaR, ent_type = ent_type, 
                                    corr = corr_sub, renyi = renyi,
                                    partition = [4*int(LAB/2),4*int(LAB/2)], 
                                    is_non_herm = is_non_herm, PT = PT)
                                       
        result_ent.update_ent(ent,i)
        
              
    if N < 17:
        # Generate the many body spectrum 
        cut_off_save = 1000
        m_eig, k_eig = Ferm.many_body(cut_off_save)
        
        # Sort takes long...
        print("--- %s seconds ---\n" % (time.time() - start_time))
        
        plt.figure(1)
        plt.grid(True)
        plt.scatter(k_eig, m_eig, s=10,color='blue')
        plt.ylim(-0.2,4.5)
    #    plt.axis(axis_range)
        plt.xlabel('$S$',fontsize=20)
        plt.ylabel('$\Delta$',fontsize=20)
        if offset == 0:
            plt.title(model_name + r' Ferm spectrum (PBC), $L=2\times %i$' % N)
        elif offset == pi/N:
            plt.title(model_name + r' Ferm spectrum (APBC), $L=2\times %i$' % N)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        fig = plt.gcf()
        plt.show()
        
                
        save_name = input('--> Input the save fig name: (press <ENTER> for not to save)')
        if save_name!="":
            fig.savefig(Dir.save_dir+save_name+'.pdf', bbox_inches='tight')
             
    
    if is_save == 1: Dir.save_data([Ferm, result_ent]) 
     
                
    return Ferm, result_ent, Dir



if __name__ == "__main__":
    is_get_spec = 1
    is_from_new = 1
    
    if is_get_spec==1:
        N = 40
        [Ferm, result, Dir] = get_spec(N = N, is_save = 1)
        
        x_data = result.NAB
        y_data = [result.SAB.real, result.MI.real,
                  result.RE.real, 3*(result.RE.real-result.MI.real)/np.log(2)]
        x_labels = ['$l_{AB}$','$l_{AB}$','$l_{AB}$','$l_{AB}$']
        y_labels = ['$S_{AB}$','$I$','$\mathcal{E}$','$h_{A:B}$']
        plot_style(x_data,y_data, Dir, N = result.N, 
                   x_labels = x_labels, y_labels = y_labels)
        
        # scipy.io.savemat('RE_MI_data.mat', dict(RE = result.RE, MI = result.MI))
        
        
    else:
        Dir = File_access()
        task = 2
            
        [Ferm, result] = Dir.get_back_ext(is_from_new)
     
        
        if task == 1:
            N = Ferm.N
            n1, n2 = 1, 1
            k1, k2 = (2*pi/N)*n1, (2*pi/N)*n2
            s1, s2 = 1, 1 # 1 is plus, -1 is minus
            n = 0
            result2 = ferm_tool.hn_element(Ferm,k1,k2,s1,s2,n)
        
        elif task == 2:
            x_data = result.NAB
            # y_data = [SA,RA,RE,(RE-MI)/(np.log(2)/3)]
            y_data = [result.SAB.real, result.RenyiAB.real,
                      result.RE.real, result.LN.real]
            x_labels = ['$l_{AB}$','$l_{AB}$','$l_{AB}$','$l_{AB}$']
            y_labels = ['$S_{AB}$','$Renyi$','$R$','$\mathcal{E}$']
            plot_style(x_data,y_data,Dir,x_labels = x_labels, y_labels = y_labels)
            
            scipy.io.savemat('RE_MI_data.mat', dict(RE = result.RE, MI = result.MI))
            
        
        # fermSSH//ferm_spec_14
        # Use Ferm.__dict__.keys() to see all the instant variables

        
    


   