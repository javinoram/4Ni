from base import *

Hz = np.linspace(0.0, 10, 501)
T = np.linspace(1,10, 101)
PhaseIso = []
for l,h in enumerate(Hz):
      print(h)
      H = hamiltoniano([1.49, 0.5, h])
      ee1, vv1= get_eigen(H)

      aux = []
      for i,t in enumerate(T):
            partition = jnp.exp( jnp.divide(-ee1, t*boltz) )
            val =  Specific_heat(jnp.array(ee1), t, partition)
            aux.append( val )
      PhaseIso.append( aux )