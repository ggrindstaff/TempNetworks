
from .utils import level_set,plot_diags,temp_net_to_clique_clx
from .topfunctions import dgm_correlation, graph_autocorrelation
import teneto
from teneto import TemporalNetwork
import dionysus as d
#from bats.visualization.plotly import ScatterVisualization
import scipy.spatial.distance as distance
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import itertools
import pandas as pd
import random
#import persim
np.random.seed(10)
from networkx.algorithms.flow.networksimplex import network_simplex
from matplotlib.animation import FuncAnimation
from matplotlib import rc
rc('animation', html='jshtml')


def to_temp_net(net,times,w):
  n = net.N
  T = np.max(times)
  adj = np.zeros(shape=(n,n,T))
  for e in net.get_edges():
    for (t1,t2) in times[e]:
      for i in len(w):
        if t1<=w[i]<=t2:
          adj[e[0],e[1],i] = 1
  TN = TemporalNetwork(from_array=adj,timelabels=w)
  return TN

def partition(tnet,w):
  df = pd.DataFrame(columns=['i','j','t','weight'])
  if tnet.timelabels!=None:
    times = tnet.timelabels
  else: times = list(range(tnet.T))
  for time in times:
    for i in range(len(w)):
      if i == len(w)-1 and time>=w[i]:
        new = tnet.get_network_when(t=time)
        new['t'] = [i for j in range(len(new))]
        df = pd.concat([df,new])
      elif w[i]<=time <w[i+1]:
        new = tnet.get_network_when(t=time)
        new['t'] = [i for j in range(len(new))]
        df = pd.concat([df,new])
  print(df)
  return TemporalNetwork(from_df=df,timelabels=w)


#discrete
tbin = TemporalNetwork()
tbin.generatenetwork('rand_binomial',size=(10,40), prob=0.3)
#tbin.get_network_when(i=0)
tbin.plot('slice_plot')
plt.savefig('./figs/binomial_slice.png')
plt.close()
#tbin.plot('circle_plot')
#tbin.plot('graphlet_stack_plot')

tpoiss = TemporalNetwork()
tpoiss.generatenetwork('rand_poisson',nnodes=10,ncontacts=4,lam=8)
tpoiss.get_network_when(i=1)

sxs, times,time_bank = temp_net_to_clique_clx(tbin)
# flag function
zz,dgms,cell= d.zigzag_homology_persistence(d.Filtration(sxs),times)

plot_diags(dgms,tbin.T,rotate=False,save='./figs/tbin_diags')
#continuous

