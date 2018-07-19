import matplotlib.pylab as plt
import numpy as np

import dm_models as dm

from time import time

target_atom = dm.AGe
target_mass = 1.0 # kg

massDM = 100.0 # GeV
sigma_n = 7e-42
elo = 0.5
ehi = 20.0

max_days = 365*1

efficiency = lambda x: 1.0

model = 'shm'

npts = 100
energy = np.linspace(elo,ehi,npts)
days = np.linspace(1,max_days,npts)

start = time()
vmin = 0
vstrE = 200
v0 = 100
vm = dm.vmin(energy, target_atom, massDM)
print(vm)
#yenergy = dm.gStream(vmin, vstrE, v0)
print("Time: %f ms" % (time()-start))

plt.figure(figsize=(10,6))

plt.subplot(3,3,1)
plt.plot(energy,vm)
plt.xlabel('Energy',fontsize=12)
plt.xlabel('Vmin',fontsize=12)


gshm = dm.gSHM(energy, days, target_atom, massDM)

plt.subplot(3,3,2)
plt.plot(energy,gshm)
plt.xlabel('Energy',fontsize=12)
plt.xlabel('gSHM',fontsize=12)

sp = dm.spectraParticle(energy, target_atom, massDM, sigma_n)

plt.subplot(3,3,3)
plt.plot(energy,sp)
plt.xlabel('Energy',fontsize=12)
plt.xlabel('spectraParticle',fontsize=12)

dr = dm.dRdErSHM(energy, days, target_atom, massDM, sigma_n)
print("dRdE")
print(dr)

plt.subplot(3,3,4)
plt.plot(energy,dr)
plt.xlabel('Energy',fontsize=12)
plt.xlabel('dR/dE SHM',fontsize=12)


plt.tight_layout()

plt.show()
