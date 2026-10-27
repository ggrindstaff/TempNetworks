#hospital

import pandas as pd
import numpy as np
import networkx as nx
import dionysus as d
import itertools
import matplotlib.pyplot as plt
from tempnets.utils import timelist_to_ee_non_int, plot_diags
from tempnets.topfunctions import graph_autocorrelation, top_autocorr, top_burstiness
#from ..tempnets.utils import 
#from ..tempnets.topfunctions import 
from pathlib import Path

current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / 'data'
data_file_path = data_dir / 'detailed_list_of_contacts_Hospital.dat'

hosp = pd.read_csv(data_file_path,delimiter = "\t",header=None)
hosp.columns = ['t','i','j','type1','type2']
hlist = hosp[['i','j','t']].values.tolist()
edgelist = hosp[['i','j']].values.tolist()
#contact times for each pair, keyed (min,max) whichever order the data lists it in
timespd = {}
for (i,j),ts in hosp.groupby(by=['i','j'])['t']:
  timespd[(min(i,j),max(i,j))] = sorted(timespd.get((min(i,j),max(i,j)),[]) + list(ts))
G = nx.Graph(edgelist)
nodes = sorted(G.nodes)

"""time is in 20-sec intervals, so one day is 4320 time units."""

day=4320
hour = 180
minute = 3
T = np.max(hosp['t'])
spread = 2*hour
all_times = list(range(0,T+20,20))
#the complex is built in units of all_times indices (20 sec), not seconds
T_idx = len(all_times)

"""We show that different buffer times result in the same topological statistics"""

buffer_list = [minute,10*minute, hour, 4*hour, 10*hour, day, 2*day]
buffer = ["1 minute","10 minutes", "1 hour", "4 hours", "10 hours", "24 hours", "48 hours"]

hosp_dgm = []
hosp_graph_auto=[]
#G with sorted nodes and (min,max) edges; the edge entry/exit lists are added in
#G_ordered.edges order, which graph_autocorrelation/level_set relies on
G_ordered = nx.Graph()
G_ordered.add_nodes_from(nodes)
G_ordered.add_edges_from(timespd.keys())
edges = set(timespd.keys())

for j in range(len(buffer_list)):
  sxs = [[i] for i in nodes]
  times = [[0,T_idx] for i in nodes]
  for e in G_ordered.edges:
    sxs.append(list(e))
    times.append(timelist_to_ee_non_int(timespd[e],all_times,buffer=buffer_list[j]))

  for sx in itertools.combinations(nodes,3):
      boundary1 = list(itertools.combinations(sx,2))
      if all([b in edges for b in boundary1]):
        time = set(timespd[boundary1[0]])
        for b in boundary1:
          time =  set.intersection(time,set(timespd[b]))
        if len(time)>0:
          sxs.append(list(sx))
          temp = sorted(list(time))
          times.append(timelist_to_ee_non_int(temp,all_times,buffer=buffer_list[j]))

  fil = d.Filtration(sxs)
  zz,dgms,cell = d.zigzag_homology_persistence(fil,times)
  hosp_graph_auto.append(graph_autocorrelation(G_ordered,times,np.linspace(0, T_idx)))
  hosp_dgm.append(dgms)
  plot_diags(dgms,T_idx,title = buffer[j],save='figs/'+buffer[j])

auto_array = [top_autocorr(hosp_dgm[i],np.linspace(0, T_idx),format='dionysus') for i in range(7)]
plt.figure()
plt.plot(auto_array)
plt.ylim(0,1)
plt.xticks([0,1,2,3,4,5,6],labels=buffer)
plt.ylabel("Topological persistence autocorrelation")
plt.xlabel("Buffer window")
plt.savefig('figs/hosp_top_auto.png')

plt.figure()
plt.plot(hosp_graph_auto)
plt.ylim(-1,1)
plt.xticks([0,1,2,3,4,5,6],labels=buffer)
plt.ylabel("Graph autocorrelation")
plt.xlabel("Buffer window")
plt.savefig('figs/hosp_graph_auto.png')

burst = [top_burstiness(dg,T_idx) for dg in hosp_dgm]
plt.figure()
plt.plot(np.transpose(burst))
plt.xticks(range(6),labels=["0-birth", "0-death","1-birth","1-death","2-birth","2-death"])
plt.legend(labels=buffer)
plt.title("Burstiness of homological birth and death times")
plt.savefig('figs/hosp_top_burst.png')