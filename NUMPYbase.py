import numpy as np
import numpy.linalg as la
import pandas as pd
import sys

"""
Seccion de funciones de observables.
Aca se definen ciertas funciones utilies para calcular observables.
Estas formulas fueron extraidas de https://www.mdpi.com/1996-1944/13/19/4399

ee: vector de valores propios (meV)
t: temperatura (K)
partition: vector de exp(-E_i / T*k_b) para cada valor propio
mag: vector de valores esperados de la magnetizacion para cada vector propio
rho: operador de densidad
spin: numero de espines del sistema
spin_val: tamaño del vector de estado de cada sitio individual (spin 0.5 -> 2, spin 1 -> 3)
"""

dtype = 'float64'
boltz = 8.617333262e-2 #mev/K
gyro = 2.0
nub = 5.7883818066e-2 #meV/T

def NumpySpecific_heat(ee, t):
    partition = np.exp( np.divide(-ee, t*boltz, dtype=dtype), dtype=dtype )
    Z = np.sum(partition, dtype=dtype)
    partition = np.divide( partition, Z, dtype=dtype )

    aux1 = np.sum(partition*ee, dtype=dtype)
    aux2 = np.sum(partition*(ee**2), dtype=dtype)
    return np.divide(aux2- (aux1**2), (t*t*boltz), dtype=dtype)


def NumpyEntropy(ee, t, partition):
    partition = np.exp( np.divide(-ee, t*boltz, dtype=dtype), dtype=dtype )
    Z = np.sum(partition, dtype=dtype)
    partition = np.divide( partition, Z, dtype=dtype )

    termal = np.sum( ee*partition, dtype=dtype)
    free_energy = -boltz*np.log( Z, dtype=dtype) 
    return np.divide(termal, t, dtype=dtype) - free_energy 

def NumpyvonNeumann(ee):
    aux = np.sum( -ee*np.log(ee, dtype=dtype), dtype=dtype )
    return aux

def NumpyPartialTraceLR(rho, spin, spin_val):
    aux = np.zeros((int(rho.shape[0]/spin_val), int(rho.shape[1]/spin_val)))
    for i in range(spin_val):
        valaux = spin_val**(spin-1)
        aux = aux + rho[valaux*i:valaux*(i+1), valaux*i:valaux*(i+1)]
    return aux

def NumpyPartialTraceRL(rho, spin, spin_val):
    aux = np.zeros((int(rho.shape[0]/spin_val), int(rho.shape[1]/spin_val)))
    for i in range(spin_val**(spin-1)):
        for j in range(spin_val**(spin-1)):
            aux[i,j] = np.trace( rho[spin_val*i:spin_val*(i+1), spin_val*j:spin_val*(j+1)] )
    return aux

def NumpyMagnetization(mag, ee, t):
    partition = np.exp( np.divide(-ee, t*boltz, dtype=dtype), dtype=dtype )
    Z = np.sum(partition, dtype=dtype)
    partition = np.divide( partition, Z, dtype=dtype )
    mag = np.sum( mag*partition, dtype=dtype)
    return mag

    
def Numpyget_eigen(H):
    ee, vv= np.linalg.eigh(H)
    return ee, vv


## Hamiltonian construction
def Numpyhamiltoniano(params):
    H = params[0]*Int1 + params[1]*Int2 -params[2]*OZ
    H = np.real( H )
    return H

Si = np.array( [ [1,0,0], [0,1,0], [0,0,1] ] ,dtype='float64')
Sx = (1.0/np.sqrt(2))*np.array( [ [0,1,0], [1,0,1], [0,1,0] ], dtype='float64') 
Sy = (1.0/np.sqrt(2))*np.array( [ [0,-1j,0], [1j,0,-1j], [0,1j,0] ], dtype='complex64') 
Sz = np.array( [ [1,0,0], [0,0,0], [0,0,-1] ], dtype='float64') 


Int1 = np.kron(np.kron(Sx, Sx), np.kron(Si, Si)) + np.kron(np.kron(Sy, Sy), np.kron(Si, Si)) + np.kron(np.kron(Sz, Sz), np.kron(Si, Si))

Int2 = np.kron(np.kron(Sx, Si), np.kron(Sx, Si)) + np.kron(np.kron(Sy, Si), np.kron(Sy, Si)) + np.kron(np.kron(Sz, Si), np.kron(Sz, Si)) +\
    np.kron(np.kron(Sx, Si), np.kron(Si, Sx)) + np.kron(np.kron(Sy, Si), np.kron(Si, Sy)) + np.kron(np.kron(Sz, Si), np.kron(Si, Sz)) +\
    np.kron(np.kron(Si, Sx), np.kron(Sx, Si)) + np.kron(np.kron(Si, Sy), np.kron(Sy, Si)) + np.kron(np.kron(Si, Sz), np.kron(Sz, Si)) +\
    np.kron(np.kron(Si, Sx), np.kron(Si, Sx)) + np.kron(np.kron(Si, Sy), np.kron(Si, Sy)) + np.kron(np.kron(Si, Sz), np.kron(Si, Sz))

OZ =  np.kron(np.kron(Sz, Si), np.kron(Si, Si)) +  np.kron(np.kron(Si, Sz), np.kron(Si, Si)) +\
    np.kron(np.kron(Si, Si), np.kron(Sz, Si)) +  np.kron(np.kron(Si, Si), np.kron(Si, Sz))