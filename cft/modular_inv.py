# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 11:13:43 2022

@author: Yuhan Liu (yuhanliu@uchicago.edu)
"""

import numpy as np
from math import pi
from scipy.linalg import eig

from toolkit.plot_style import plot_style_s

N_DEFAULT = 2000


def theta_2(tau, N=N_DEFAULT):
    t = 0
    
    for n in range(-N, N+1, 1):
        t = t+np.exp(2*pi*1j*tau*(1/2*(n+1/2)**2))
        
    return t


def theta_3(tau, N=2000):
    t = 0
    for n in range(-N, N+1, 1):
        t = t+np.exp(2*pi*1j*tau*(1/2*n**2))
        
    return t


def theta_4(tau, N=2000):
    t = 0
    for n in range(-N, N+1, 1):
        t = t+np.exp(2*pi*1j*tau*(1/2*n**2))*(-1)**n
        
    return t


def eta(tau, N = 2000):
    q = np.exp(2*pi*1j*tau)
    
    t = np.exp(2*pi*1j*tau/24)
    for n in range(1, N):
        t = t*(1-q**n)
        
    return t


def Theta(tau, m, k, N = 2000):
    t = 0
    q = np.exp(2*pi*1j*tau)
    
    for n in range(-N, N+1, 1):
        t = t+q**(k*(n+m/(2*k))**2)
    
    return t


def Theta_e(tau, m, k, N = 2000): return Theta(tau,m,k,N)/eta(tau,N)

def Theta_1(tau, z, N = N_DEFAULT):
# theta_1/eta
    
    qfac = np.exp(2*pi*1j*tau*(1/8-1/24))
    yfac = np.exp(pi*1j*z)

    q = np.exp(2*pi*1j*tau)
    y = np.exp(2*pi*1j*z)
    
    t = -1j*yfac*qfac
    
    for n in range(0, N+1, 1):
        t = t*(1-y*q**(n+1))*(1-(1/y)*q**n)
    
    return t


def theta_2e(tau, N = 2000): return theta_2(tau,N)/eta(tau, N)
def theta_3e(tau, N = 2000): return theta_3(tau,N)/eta(tau, N)
def theta_4e(tau, N = 2000): return theta_4(tau,N)/eta(tau, N)

def chi_0(tau, N = 2000): return 1/2*(np.sqrt(theta_3e(tau,N))+np.sqrt(theta_4e(tau,N)))
def chi_ep(tau, N = 2000): return 1/2*(np.sqrt(theta_3e(tau,N))-np.sqrt(theta_4e(tau,N)))
def chi_sigma(tau, N = 2000): return 1/np.sqrt(2)*np.sqrt(theta_2e(tau,N))

def chi_orb_I(tau,k, N = 2000):
    q = np.exp(1j*2*pi*tau)
    
    t = 0
    for n in range(-N,N+1,1):
        t = t+q**(k*n**2)+(-1)**n*q**(n**2)
        
    t = t/(2*eta(tau, N))
    
    return t


def chi_orb_Theta(tau,k, N = 2000):
    q = np.exp(1j*2*pi*tau)
    
    t = 0
    for n in range(-N,N+1,1):
        t = t+q**(k*n**2)-(-1)**n*q**(n**2)
        
    t = t/(2*eta(tau, N))
    
    return t


def check_dedekind(tau, N = 2000):
    # Example: tau = 2.25*1j
    # If we choose tau to be real number, the error would
    # increase as we increase N.
    
    errt_1 = theta_2(tau+1, N)-np.exp(1j*pi/4)*theta_2(tau,N)
    errt_2 = theta_3(tau+1, N)-theta_4(tau,N)
    errt_3 = theta_4(tau+1, N)-theta_3(tau,N)
    
    print("error T: ", abs(errt_1))
    print("error T: ", abs(errt_2))
    print("error T: ", abs(errt_3))
    
    
    errs_1 = theta_2(-1/tau, N)/eta(-1/tau, N)-theta_4(tau, N)/eta(tau, N)
    errs_2 = theta_3(-1/tau, N)/eta(-1/tau, N)-theta_3(tau, N)/eta(tau, N)
    errs_3 = theta_4(-1/tau, N)/eta(-1/tau, N)-theta_2(tau, N)/eta(tau, N)
    
    print("error S: ", errs_1)
    print("error S: ", errs_2)
    print("error S: ", errs_3)
    
    return 0


def check_convergence(tau):
    N_tot = range(1000,8000,500)
    t2 = np.zeros(len(N_tot), dtype = complex)

    for i in range(len(N_tot)):
        t2[i] = theta_2(tau, N_tot[i])
        
    plot_style_s(N_tot, abs(t2))


def Z_Ising(tau, N = 2000):
    
    Z = 0.5*(abs(theta_2e(tau, N)) + abs(theta_3e(tau, N)) \
        + abs(theta_4e(tau, N)))
        
    return Z


def Z_u1k(tau, k, N=2000):
    Z = 0
    for m in range(-k+1, k+1,1):
        Z = Z+abs(Theta_e(tau, m, k, N))**2

    return Z


def Zc2(tau, N = N_DEFAULT):
    
    t = 0
    q = np.exp(1j*2*pi*tau)
    
    for m1 in range(-N, N+1,1):
        for m2 in range(-N,N+1,1):
            for n1 in range(-N, N+1, 1):
                for n2 in range(-N,N+1,1):
                    p1 = 1/np.sqrt(6)*(n2*np.sqrt(3)/2+2*np.sqrt(3)*m2+np.sqrt(3)*m1)
                    pbar1 = 1/np.sqrt(6)*(n2*np.sqrt(3)/2-2*np.sqrt(3)*m2-np.sqrt(3)*m1)
                    p2 = 1/np.sqrt(6)*(-n2/2+n1+3*m1)
                    pbar2 = 1/np.sqrt(6)*(-n2/2+n1-3*m1)
                    
                    t = t+q**(1/2*(p1**2+p2**2))*q**(1/2*(pbar1**2+pbar2**2))


    t = t/(abs(eta(tau,N)))**4
    
    return t

def Z_boson_orb(tau, k, N = 2000):
    
    """
    Benchmark: Z_boson_orb(tau,2) = Z_Ising(tau)**2
    """
    
    Z = 0.5*(Z_u1k(tau, k, N)+2*Z_u1k(tau,4,N)-Z_u1k(tau,1,N))
    
    return Z
    


def Z_Ising_orb(tau, N = 2000):
    """
    Benchmark: Z_Ising_orb(tau) = Z_boson_orb(tau,8)
    """
    Z = 0.5*Z_Ising(tau, N)**2 + 0.5*Z_Ising(2*tau, N)+0.5*Z_Ising(tau/2, N) \
        +0.5*Z_Ising(tau/2+1/2, N)
        
    return Z 


def Z3_orb(tau, N = N_DEFAULT):
    q = np.exp(2*pi*1j*tau)
    
    Z = abs(1/Theta_1(tau,1/3,N))**2+abs(q)**(-1/9)*(
        abs(1/Theta_1(tau,tau/3,N))**2+abs(1/Theta_1(tau,tau/3+1/3,N))**2
        +abs(1/Theta_1(tau,tau/3+2/3,N))**2)
    
    return Z


def check_Ising_orb_corrs(tau, N = 2000):
    """
    Check the correspondence between Ising orbifold and free boson orbirold (k=8)
    There are 15 primary fields
    """
    
    phi_2 = Theta_e(tau,2,8)
    phi_2I = chi_sigma(tau)**2/2+chi_sigma(2*tau)/2
    
    phi_4 = Theta_e(tau, 4, 8)
    phi_4I = chi_0(tau)*chi_ep(tau)
    
    phi_6 = Theta_e(tau, 6, 8)
    phi_6I = chi_sigma(tau)**2/2-chi_sigma(2*tau)/2
    
    phi_1 = Theta_e(tau,1,8)
    phi_1I = 1/2*(chi_0(tau/2)+chi_0(tau/2+1/2)*np.exp(1j*pi/48))
    
    phi_3 = Theta_e(tau,3,8)
    phi_3I = 1/2*(chi_ep(tau/2)+chi_ep(tau/2+1/2)*np.exp(1j*pi*(1/48-1/2)))
    
    phi_5 = Theta_e(tau,5,8)
    phi_5I = 1/2*(chi_ep(tau/2)-chi_ep(tau/2+1/2)*np.exp(1j*pi*(1/48-1/2)))
    
    phi_7 = Theta_e(tau,7,8)
    phi_7I = 1/2*(chi_0(tau/2)-chi_0(tau/2+1/2)*np.exp(1j*pi/48))
    
    sg = Theta_e(tau, 1,4)
    sg_I = 1/2*(chi_sigma(tau/2)+chi_sigma(tau/2+1/2)*np.exp(1j*pi*(1/48-1/16)))
    sg_II = chi_0(tau)*chi_sigma(tau)
    
    Tau = Theta_e(tau, 3,4)
    Tau_I = 1/2*(chi_sigma(tau/2)-chi_sigma(tau/2+1/2)*np.exp(1j*pi*(1/48-1/16)))
    Tau_II = chi_sigma(tau)*chi_ep(tau)
    
    Phi_1 = Theta_e(tau, 8, 8)/2
    Phi_1I = chi_0(tau)**2/2-chi_0(2*tau)/2
    Phi_1II = chi_ep(tau)**2/2-chi_ep(2*tau)/2
    
    orbI = chi_orb_I(tau, 8)
    orbI_I = chi_0(tau)**2/2+chi_0(2*tau)/2
    
    orbTheta = chi_orb_Theta(tau, 8)
    orbTheta_I = chi_ep(tau)**2/2+chi_ep(2*tau)/2
    
    print(phi_7)
    print(phi_7I)
    
    return 0


def check_Z3_orb_inv(tau):
    
    ## Note: theta_1(z+1,tau) = -theta_1(z,tau)
    ##       theta_1(-z,tau) = -theta_1(z,tau)
    
    unt = lambda tau: Theta_1(tau,1/3)
    tw1 = lambda tau : Theta_1(tau,tau/3)*np.exp(1j*pi*tau/9)
    tw2 = lambda tau : Theta_1(tau,tau/3+1/3)*np.exp(1j*pi*tau/9)
    tw3 = lambda tau : Theta_1(tau,tau/3+2/3)*np.exp(1j*pi*tau/9)
  
    unt_T = unt(tau+1)-unt(tau)*np.exp(1j*(pi/4-pi/12))
    tw1_T = tw1(tau+1)-tw2(tau)*np.exp(1j*(pi/4-pi/12+pi/9))
    tw2_T = tw2(tau+1)-tw3(tau)*np.exp(1j*(pi/4-pi/12+pi/9))
    tw3_T = tw3(tau+1)-tw1(tau)*np.exp(1j*(pi/4-pi/12+pi/9))*(-1)
  
    unt_S = unt(-1/tau) - tw1(tau)*(-1j)   
    tw1_S = tw1(-1/tau) - (-1j*Theta_1(tau,-1/3))
    tw1_S = tw1(-1/tau) - (1j*unt(tau))
    tw2_S = tw2(-1/tau) - tw3(tau)*(1j)*np.exp(-1j*pi*2/9)
    tw3_S = tw3(-1/tau) - tw2(tau)*(-1j)*np.exp(1j*pi*2/9)

    print(tw1_S)
    print(tw1_T)
    
    return 0


tau = 0.23*1j
check_Z3_orb_inv(tau)

test = np.array([[0,1],[np.exp(pi*1j/24),0]])
test = np.array([[0, np.exp(1j*5*pi/18),0] , 
                [0,0, np.exp(1j*5*pi/18)],
                [ -np.exp(1j*5*pi/18), 0,0]])

t1, t2 = eig(test)

    