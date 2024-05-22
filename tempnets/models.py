
from .utils import *
#level_set,plot_diags,temp_net_to_clique_clx, ER_periodic, to_ee, is_in, exponential_rewiring, is_in_ee,intersection
from .topfunctions import graph_autocorrelation, top_autocorr, burstiness, pers_burstiness
from .teneto_utils import sun_array
from teneto import TemporalNetwork
import dionysus as d
#from bats.visualization.plotly import ScatterVisualization
import scipy.spatial.distance as distance
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
#import persim
np.random.seed(10)
from networkx.algorithms.flow.networksimplex import network_simplex
from matplotlib import rc
from math import comb
rc('animation', html='jshtml')


#global parameters
T = 100
n = 40
max_dim=2

#expected number of edges / (n choose 2) at each time point
#e.g. p for the Erdos-Renyi models
dens = 0.3

#we also specify the temporal rate of edge formation
new_edge_rate = 0.1

#store results
diags = []
adjacency = []
topo = []
bursty_cv = []
topo_cv = []


#discrete
tbin = TemporalNetwork()
q = dens + new_edge_rate - new_edge_rate/dens
tbin.generatenetwork('rand_binomial',size=(n,T), prob=[new_edge_rate,q])
dgmsbin = teneto_homology(tbin)
diags.append(dgmsbin)

tpoiss = TemporalNetwork()
tpoiss.generatenetwork('rand_poisson',nnodes=n,ncontacts=int(dens*T),lam=int(1/dens))
#tpoiss.get_network_when(i=1)
dgmspoiss = teneto_homology(tpoiss)
diags.append(dgmspoiss)


#Erdos-Renyi params
period = 1/new_edge_rate
l = 1/q
p = dens*(period)/l #probability of edges
#periodic params
#freq = 8 #time between appearances
#l = 2   #how long edges stick around

#Choose edges with probability $p$, then if chosen, generate a random periodic sequence of timestamps

netER,sxsER,timesER = ER_periodic(n,p,period-l,l,T,max_dim)

Y = d.Filtration(sxsER)
dtimes = to_ee(timesER,sxsER,T)
zzER,dgmsER,cellER=d.zigzag_homology_persistence(Y,dtimes)
diags.append(dgmsER)


# Watts-Strogatz with exponential rewiring

k= 3
p = 0
l = 2


G = nx.watts_strogatz_graph(n,k,p)
netWS,sxsWS,timesws = exponential_rewiring(G,l,T,max_dim)
Y = d.Filtration(sxsWS)
dtimes=to_ee(timesws,sxsWS,T)

zz,dgmsWS,cell=d.zigzag_homology_persistence(Y,dtimes)
diags.append(dgmsWS)




#num rays
r = 5
#length of rays
l = n/r
#rounds
#k = int(T/l)
#T = int(l * k)
tsun = TemporalNetwork(from_array=sun_array(n,T,l))
sxsSun, timesSun = temp_net_to_clique_clx(tsun)
zzSun,dgmsSun,cell= d.zigzag_homology_persistence(d.Filtration(sxsSun),timesSun)
diags.append(dgmsSun)

#dictimes = {tuple(i):j for i,j in zip(sxs,times)}



#Here we have a random graph (uniform on edges) that is topologically self-inhibiting.

l = 1/new_edge_rate
G = nx.gnp_random_graph(n,dens)

netTSI,sxsTSI,timesTSI = topological_rewiring(G,l,T)
zzTSI,dgmsTSI,cellTSI= d.zigzag_homology_persistence(d.Filtration(sxsTSI),to_ee(timesTSI,sxsTSI,T))
diags.append(dgmsTSI)


#Plotting 

plot_diags(dgmsbin,tbin.T,rotate=False,title='Binary network',save='figs/tbin_diags')
plot_diags(dgmspoiss,tpoiss.T,rotate=False,title = 'Poisson network', save='figs/tpoiss_diags')
plot_diags(dgmsER,T,rotate=False,save = 'figs/ER_diags')
plot_diags(dgmsWS,T,save = 'figs/WS_diags')
plot_diags(dgmsSun,T,save = './figs/sun_diag')
plot_diags(dgmsTSI,T, save = 'figs/TSI_diags')

# Comparison Tests

#Comparing adjacency and topological autocorrelation



for i in range(50):
  net,X,times = ER_periodic(n,p,period-l,l,T,max_dim)
  bursty_cv.append(burstiness(times))
  #all_K = X.complex().get_simplices()
  zz,dgms,cell=d.zigzag_homology_persistence(d.Filtration(X),to_ee(times,X,T))
  #cv(all_finite_pers[0]+all_finite_pers[1])
  topo_cv.append(pers_burstiness(dgms,T))
  adjacency.append(graph_autocorrelation(net,dtimes,np.linspace(0, T, T)))
  topo.append(top_autocorr(dgms,np.linspace(0, T, T),format='dionysus'))

plt.scatter(adjacency,topo)
plt.xlabel("Graph adjacency correlation")
plt.ylabel("Topological correlation")
plt.title("Experiment 1, ER periodic")
plt.savefig('./figs/adjacency.png')

# Compare burstiness and topological burstiness

plt.scatter(bursty_cv,topo_cv)
plt.xlabel("Burstiness (CV)")
plt.ylabel("Topological variation")
plt.title("Experiment 1, ER periodic")
plt.legend()
plt.savefig('./figs/burstiness.png')

"""
print(graph_autocorrelation(netER,dtimes,np.linspace(0, T, T)))
print(top_autocorr(dgmsER,np.linspace(0, T, T),format='dionysus'))
"""