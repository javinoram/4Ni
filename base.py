## Imports
import numpy as np
import jax
import jax.numpy as jnp

## Magnetocaloric funcions and other things

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

#Constantes importantes, todo esta en milielectronvolt
dtype = 'float64'
boltz = 8.617333262e-2 #mev/K
gyro = 2.0
nub = 5.7883818066e-2 #meV/T

@jax.jit
def Specific_heat(ee, t, partition):
    Z = jnp.sum(partition)
    aux1 = jnp.divide( jnp.sum( partition*ee ), Z )
    aux2 = jnp.divide( jnp.sum( partition*(ee**2) ) , Z )
    return jnp.divide( aux2- (aux1**2), (t*t*boltz) )

@jax.jit
def Entropy(ee, t, partition):
    Z = jnp.sum( partition )
    partition = np.divide( partition, Z )
    termal = jnp.sum( partition*ee )
    free_energy = -boltz*jnp.log( Z ) 
    return jnp.divide(termal, t ) - free_energy 

@jax.jit
def vonNeumann(ee):
    aux = jnp.sum( -ee*( jnp.log(ee ) ) )
    return aux

@jax.jit
def Magnetization(mag, partition):
    Z = jnp.sum( partition )
    partition = jnp.divide( partition, Z )    
    mag = jnp.sum( partition*mag )
    return mag

@jax.jit
def get_eigen(H):
    ee, vv= jnp.linalg.eigh(H)
    return ee, vv




## Hamiltonian construction

def hamiltoniano(params):
    H = params[0]*Int1 + params[1]*Int2 -params[2]*OZ
    H = np.real( H, dtype=dtype )
    return jnp.array(H)


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