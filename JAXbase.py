## Imports
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

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
def JAXSpecific_heat(ee, t):
    partition = jnp.exp( jnp.divide(-ee, t*boltz) )
    Z = jnp.sum(partition)
    partition = jnp.divide(partition, Z)
    
    aux1 = jnp.sum( partition*ee )
    aux2 = jnp.sum( partition*(ee**2) )
    return jnp.divide( aux2- (aux1**2), (t*t*boltz) )
vec_JAXSpecific_heat = jax.vmap(JAXSpecific_heat, in_axes=(None, 0))

@jax.jit
def JAXEntropy(ee, t):
    partition = jnp.exp( jnp.divide(-ee, t*boltz) )
    Z = jnp.sum( partition )
    partition = np.divide( partition, Z )

    termal = jnp.sum( partition*ee )
    free_energy = -boltz*jnp.log( Z ) 
    return jnp.divide( termal, t ) - free_energy 
vec_JAXEntropy = jax.vmap(JAXEntropy, in_axes=(None, 0))

@jax.jit
def JAXvonNeumann(ee):
    aux = jnp.sum( -ee*( jnp.log(ee ) ) )
    return aux

@jax.jit
def JAXMagnetization(mag, ee, t):
    partition = jnp.exp( jnp.divide(-ee, t*boltz) )
    Z = jnp.sum( partition )
    partition = jnp.divide( partition, Z )    
    mag = jnp.sum( partition*mag )
    return mag
vec_JAXMagnetization = jax.vmap(JAXMagnetization, in_axes=(None, None, 0))

@jax.jit
def JAXget_eigen(H):
    ee, vv= jnp.linalg.eigh(H)
    return ee, vv




## Hamiltonian construction
@jax.jit
def JAXhamiltoniano(params):
    H = params[0]*Int1 + params[1]*Int2 -params[2]*OZ
    H = jnp.real( H )
    return H


Si = jnp.array( [ [1,0,0], [0,1,0], [0,0,1] ] ,dtype='float64')
Sx = (1.0/jnp.sqrt(2))*jnp.array( [ [0,1,0], [1,0,1], [0,1,0] ], dtype='float64') 
Sy = (1.0/jnp.sqrt(2))*jnp.array( [ [0,-1j,0], [1j,0,-1j], [0,1j,0] ], dtype='complex64') 
Sz = jnp.array( [ [1,0,0], [0,0,0], [0,0,-1] ], dtype='float64') 

Int1 = jnp.kron(jnp.kron(Sx, Sx), jnp.kron(Si, Si)) + jnp.kron(jnp.kron(Sy, Sy), jnp.kron(Si, Si)) + jnp.kron(jnp.kron(Sz, Sz), jnp.kron(Si, Si))

Int2 = jnp.kron(jnp.kron(Sx, Si), jnp.kron(Sx, Si)) + jnp.kron(jnp.kron(Sy, Si), jnp.kron(Sy, Si)) + jnp.kron(jnp.kron(Sz, Si), jnp.kron(Sz, Si)) +\
    jnp.kron(jnp.kron(Sx, Si), jnp.kron(Si, Sx)) + jnp.kron(jnp.kron(Sy, Si), jnp.kron(Si, Sy)) + jnp.kron(jnp.kron(Sz, Si), jnp.kron(Si, Sz)) +\
    jnp.kron(jnp.kron(Si, Sx), jnp.kron(Sx, Si)) + jnp.kron(jnp.kron(Si, Sy), jnp.kron(Sy, Si)) + jnp.kron(jnp.kron(Si, Sz), jnp.kron(Sz, Si)) +\
    jnp.kron(jnp.kron(Si, Sx), jnp.kron(Si, Sx)) + jnp.kron(jnp.kron(Si, Sy), jnp.kron(Si, Sy)) + jnp.kron(jnp.kron(Si, Sz), jnp.kron(Si, Sz))

OZ =  jnp.kron(jnp.kron(Sz, Si), jnp.kron(Si, Si)) +  jnp.kron(jnp.kron(Si, Sz), jnp.kron(Si, Si)) +\
    jnp.kron(jnp.kron(Si, Si), jnp.kron(Sz, Si)) +  jnp.kron(jnp.kron(Si, Si), jnp.kron(Si, Sz))