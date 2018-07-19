import matplotlib.pylab as plt
import numpy as np

import dm_models as dm

from time import time

target_atom = dm.AGe
target_mass = 1.0

massDM = 10.0 # GeV
sigma_n = 7e-42
elo = 0.5
ehi = 10.0

max_days = 365*1

efficiency = lambda x: 1.0

model = 'shm'

energy = np.linspace(elo,ehi,1000)
days = np.linspace(1,max_days,1000)

start = time()
yenergy = dm.plot_wimp_energy(energy,target_atom=dm.AGe,massDM=massDM,sigma_n=sigma_n,time_range=[1,max_days],model=model)
print("Time for plotting energy distribution: %f ms" % (time()-start))

#start = time()
#ytime = dm.plot_wimp_day(days,target_atom=dm.AGe,massDM=massDM,sigma_n=sigma_n,e_range=[elo,ehi],model=model)
#print("Time for plotting time distribution: %f ms" % (time()-start))

plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
plt.plot(energy,yenergy)
plt.xlabel('Energy',fontsize=24)

#plt.subplot(1,2,2)
#plt.plot(energy,ytime)
#plt.ylim(0,1.3*max(ytime))
#plt.xlabel('Days',fontsize=24)

plt.tight_layout()

plt.show()
