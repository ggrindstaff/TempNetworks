"""Experiments on the random models (an experiment script, not a library module).

Generates each model with T = 100, n = 40, density 0.3 and new-edge rate 0.1,
prints the density and rate each achieves, computes their zigzag persistence and
saves the diagrams to figs/. It then compares graph and topological
autocorrelation, and burstiness, over 50 ER periodic networks
(figs/adjacency.png, figs/burstiness.png).

Run from the repository root with: python -m tempnets.models  (about 15 minutes)
"""
from .utils import *
#level_set,plot_diags,temp_net_to_clique_clx, ER_periodic, to_ee, is_in, exponential_rewiring, is_in_ee,intersection
from .topfunctions import graph_autocorrelation, top_autocorr, burstiness, pers_burstiness
from .discrete import sun_array, rand_binomial, rand_poisson
from .network import TemporalNetwork
from .topfunctions import top_autocorr_continuous
from .params import binomial_params, poisson_params, er_periodic_params, ws_params, rewiring_interval, interval_stats, array_stats
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

#Each random model is specified by T, the average density and the average rate of
#new edge generation; its own parameters are computed (tempnets/params.py) to match.

#average density: expected fraction of the (n choose 2) node pairs that are edges at each time
dens = 0.3

#average rate of new edge generation: expected number of new edges per node pair per unit
#time, i.e. new_edge_rate*(n choose 2) new edges per time step. Edges then last
#dens/new_edge_rate time steps on average.
new_edge_rate = 0.1

#store results
diags = []
adjacency = []
topo = []
bursty_cv = []
topo_cv = []

def report(name,stats):
  print("%s: density %.3f (target %.3f), new edge rate %.3f (target %.3f)" % (name,stats[0],dens,stats[1],new_edge_rate))


#discrete
birth, stay = binomial_params(dens,new_edge_rate)
tbin = rand_binomial(n,T,birth,stay,init=dens)
report("Binomial",array_stats(tbin))
netbin = TemporalNetwork.from_array(tbin)
dgmsbin = netbin.flag(max_dim).persistence()
diags.append(dgmsbin)

#contacts last one time step, so only the density can be matched (see poisson_params)
tpoiss = rand_poisson(n,T,poisson_params(dens))
report("Poisson",array_stats(tpoiss))
netpoiss = TemporalNetwork.from_array(tpoiss)
dgmspoiss = netpoiss.flag(max_dim).persistence()
diags.append(dgmspoiss)


#Erdos-Renyi periodic: each edge of a G(n,p_ER) graph is present for l_ER time steps
#out of every freq + l_ER. p_ER is free as long as dens < p_ER <= 1; with p_ER = 2*dens
#edges are present half of the time.
p_ER = 2*dens
freq, l_ER = er_periodic_params(dens,new_edge_rate,p_ER)

netER,sxsER,timesER = ER_periodic(n,p_ER,freq,l_ER,T,max_dim)
report("ER periodic",interval_stats(timesER,n,T))

Y = d.Filtration(sxsER)
dtimes = to_ee(timesER,sxsER,T)
zzER,dgmsER,cellER=zigzag_persistence(Y,dtimes)
diags.append(dgmsER)


# Watts-Strogatz with exponential rewiring

#rewiring keeps n*k/2 edges, so k sets the density and the time between rewirings the rate
k, l_WS = ws_params(n,dens,new_edge_rate)

G = nx.watts_strogatz_graph(n,k,0)
netWS,sxsWS,timesws = exponential_rewiring(G,l_WS,T,max_dim)
report("Watts-Strogatz rewiring",interval_stats(timesws,n,T))
Y = d.Filtration(sxsWS)
dtimes=to_ee(timesws,sxsWS,T)

zz,dgmsWS,cell=zigzag_persistence(Y,dtimes)
diags.append(dgmsWS)




#the sun network is deterministic, so it is not matched to dens/new_edge_rate
#num rays
r = 5
#length of rays
l = n/r
#rounds
#k = int(T/l)
#T = int(l * k)
tsun = sun_array(n,T,l)
netsun = TemporalNetwork.from_array(tsun)
dgmsSun = netsun.flag(max_dim).persistence()
diags.append(dgmsSun)

#dictimes = {tuple(i):j for i,j in zip(sxs,times)}



#Here we have a random graph (uniform on edges) that is topologically self-inhibiting.

G = nx.gnp_random_graph(n,dens)

netTSI,sxsTSI,timesTSI = topological_rewiring(G,rewiring_interval(n,new_edge_rate),T)
report("Topologically self-inhibiting",interval_stats(timesTSI,n,T))
zzTSI,dgmsTSI,cellTSI= zigzag_persistence(sxsTSI,to_ee(timesTSI,sxsTSI,T))
diags.append(dgmsTSI)


#Plotting 

plot_diags(dgmsbin,netbin.T,rotate=False,title='Binary network',save='figs/tbin_diags')
plot_diags(dgmspoiss,netpoiss.T,rotate=False,title = 'Poisson network', save='figs/tpoiss_diags')
plot_diags(dgmsER,T,rotate=False,save = 'figs/ER_diags')
plot_diags(dgmsWS,T,save = 'figs/WS_diags')
plot_diags(dgmsSun,netsun.T,save = './figs/sun_diag')
plot_diags(dgmsTSI,T, save = 'figs/TSI_diags')

# Comparison Tests

#Comparing adjacency and topological autocorrelation



#lag for the autocorrelation statistics: the spacing of T evenly spaced snapshots on [0, T]
tau = T/(T-1)

for i in range(50):
  net,X,times = ER_periodic(n,p_ER,freq,l_ER,T,max_dim)
  bursty_cv.append(burstiness(times))
  #all_K = X.complex().get_simplices()
  zz,dgms,cell=zigzag_persistence(X,to_ee(times,X,T))
  #cv(all_finite_pers[0]+all_finite_pers[1])
  topo_cv.append(pers_burstiness(dgms,T))
  edges = {e: ints for e, ints in times.items() if len(e)==2}
  adjacency.append(TemporalNetwork.from_intervals(edges,T,nodes=range(n)).graph_autocorrelation_continuous(tau))
  topo.append(top_autocorr_continuous(dgms,tau,T))

plt.figure()
plt.scatter(adjacency,topo)
plt.xlabel("Graph adjacency correlation")
plt.ylabel("Topological correlation")
plt.title("Experiment 1, ER periodic")
plt.savefig('./figs/adjacency.png')

# Compare burstiness and topological burstiness

plt.figure()
plt.scatter(bursty_cv,topo_cv)
plt.xlabel("Burstiness (CV)")
plt.ylabel("Topological variation")
plt.title("Experiment 1, ER periodic")
plt.savefig('./figs/burstiness.png')

"""
print(graph_autocorrelation(netER,dtimes,np.linspace(0, T, T)))
print(top_autocorr(dgmsER,np.linspace(0, T, T),format='dionysus'))
"""