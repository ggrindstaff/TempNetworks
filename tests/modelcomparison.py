from tempnets.classes import TemporalGraphComplex
import numpy as np
import matplotlib.pyplot as plt

complexes = []
trials = 10
for i in range(trials*2):
  temp = TemporalGraphComplex(n=20)
  temp.set_periodic(freq=3,l=2)
  temp.set_ER(p=0.5)
  temp.compute_homology()
  complexes.append(temp)

for i in range(trials*2):
  temp = TemporalGraphComplex(n=20)
  temp.set_periodic(freq = 8,l=2)
  temp.set_ER(p=0.5)
  temp.compute_homology()
  complexes.append(temp)

import persim

n = len(complexes)
inds = range(n)
corrdist = [[persim.wasserstein(np.transpose(np.array([complexes[key].birth[0],complexes[key].death[0]])),
                                np.transpose(np.array([complexes[key1].birth[0],complexes[key1].death[0]])))/
                                 (persim.wasserstein([],np.transpose(np.array([complexes[key].birth[0],complexes[key].death[0]]))*
                                  persim.wasserstein([],np.transpose(np.array([complexes[key1].birth[0],complexes[key1].death[0]])))))
                                 for key1 in inds] for key in inds]

c = 0

#fig,ax1 = plt.subplots(1,n,figsize = (15,5))
#plotspace = range(n)
#for i in range(len(corrdist)):
#  ax = ax1[plotspace[i]]
  # ax.scatter(trial_names,wassdist[i],
  #            label = lab[:-1]+'% trial '+lab[-1],
  #            s = 150,edgecolor ='k' )
  # ax.legend()
  # ax.set_ylabel('Wassterstein Distance')
#  c+=1
# plt.show()

#%matplotlib inline
fig,ax0 = plt.subplots(1, figsize = (6,6))
ax=ax0
ec = ax.imshow(corrdist,cmap='BrBG')
#ax.set_xticks(list(range(n)))
#ax.set_yticks(list(range(n)))
#ax.set_xticklabels(labels)
#ax.set_yticklabels(labels)
#ax.set_xticklabels(labels, Rotation=45)
cbar = plt.colorbar(ec)
ax.set_title('Wasserstein distance')
plt.tight_layout()
plt.show()