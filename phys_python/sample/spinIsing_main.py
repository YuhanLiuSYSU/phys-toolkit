import numpy as np
import time
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


# from spin_task import spin_task
from toolkit.file_io import File_access
from toolkit.plot_style import plot_style_single
from hamiltonian.spin_tool import Spin_hamiltonian
from entangle.ent_many_body import get_ent_many_total


Sx = np.array(([0, 1], [1, 0]), dtype=np.complex128)
Sy = np.array(([0, -1j], [1j, 0]), dtype=np.complex128)
Sz = np.array(([1, 0], [0, -1]), dtype=np.complex128)


def get_spectrum(N=10, Jx=-1.0, Jz=-1.0, is_plot=0,knum=50,is_save=1):
    start_time = time.time()
    Dir = File_access()
       
 #   couple_off = [[np.hstack((np.repeat([Jx],N-1),[0])),Sx,Sx,np.full(N,1)]]
    couple_off = [[np.repeat([Jx],N),Sx,Sx,np.full(N,1)]]
    couple_diag = [[np.repeat([Jz],N),Sz]]
    bands = 1
    PBC = 1
    
    Ising = Spin_hamiltonian(N, couple_diag, couple_off, PBC, bands=bands)
    Ising.label = 'hermitian'

    # Solve the eigensyste
    eigval, eigvecs, S = Ising.sort_P(knum)   
       
    # Find the overlap with the lowest state |I>
    c, a, Delta, combine = Ising.get_c()

    #-----Output-----
    print("--- %s seconds ---\n" % (time.time() - start_time))    
    print("central chage is: %f" % c)

    # Make plot
    if is_plot == 1:
        
        plt.plot(S.real, (Delta-min(Delta)).real, 'o', color='black')
        
        fig = plot_style_single(plt, x_labels = "$S$", y_labels = "$\Delta$",
                                x_lim = [-4-0.1,4+0.1], y_lim = [0-0.1,4],
                                title = '$L=$ %i'% N)
        
        if is_save == 1: Dir.save_fig(fig)
    
    # Save the variables
    if is_save == 1: Dir.save_data(Ising)
            
    return Ising
    


if __name__ == "__main__":
   
    is_get_spectrum = 1
    is_from_new = 1
        
    if is_get_spectrum:   
        N = 12
        Jx = -1.0
        Jz = -1.0
        knum = 60
        Ising = get_spectrum(N=N, Jx=Jx,Jz=Jz,plot=1,knum = knum, is_save = 0)
        
    else:
        Dir = File_access()
        Ising = Dir.get_back("Ising16")
        # Use Ising.__dict__.keys() to see all the instant variables
        
        #result = do_task(Ising, task=1, n_total=[2,-2])
        int_tot, ent_tot, coeffs = get_ent_many_total(Ising, method ='usual', 
                                                      level=0, renyi=1)
        

    # BUG: WHAT WE GET AS l_M HERE IS ACTUALLY l_MBAR
        
#----------------------------------------------------#
#   L   |   run time                |       c
#----------------------------------------------------#
#   16  |    14 secs                |   0.509748
#   17  |    32 secs                |
#   18  |    74 secs (with error!!) |
#   20  |                           |  
#----------------------------------------------------#