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
timespd = hosp.groupby(by=['i','j'])['t'].apply(list)
G = nx.Graph(edgelist)

"""time is in 20-sec intervals, so one day is 4320 time units."""

day=4320
hour = 180
minute = 3
T = np.max(hosp['t'])
spread = 2*hour
all_times = list(range(0,T+20,20))

"""We show that different buffer times result in the same topological statistics"""

buffer_list = [minute,10*minute, hour, 4*hour, 10*hour, day, 2*day]
buffer = ["1 minute","10 minutes", "1 hour", "4 hours", "10 hours", "24 hours", "48 hours"]

hosp_dgm = []
hosp_graph_auto=[]
for j in range(len(buffer_list)):
  sxs = [[i] for i in G.nodes]
  timelist = [list(range(0,T,20)) for i in G.nodes]
  times = [[0,T] for i in G.nodes]
  for a,b in G.edges:
    if a < b:
      sxs.append([a,b])
    else:
      sxs.append([b,a])
    try:
      temp = timespd[a,b]
      timelist.append(temp)
      times.append(timelist_to_ee_non_int(temp,all_times,buffer=buffer_list[j]))
    except:
      temp = timespd[b,a]
      timelist.append(temp)
      times.append(timelist_to_ee_non_int(temp,all_times,buffer=buffer_list[j]))

  for sx in itertools.combinations(G.nodes,3):
      boundary1 = list(itertools.combinations(sx,2))
      if all([list(b) in sxs for b in boundary1]):
        time = set(timespd[boundary1[0]])
        for b in boundary1:
          time =  set.intersection(time,set(timespd[b]))
        if len(time)>0:
          sxs.append(list(sx))
          temp = sorted(list(time))
          timelist.append(temp)
          times.append(timelist_to_ee_non_int(temp,all_times,buffer=buffer_list[j]))

  fil = d.Filtration(sxs)
  zz,dgms,cell = d.zigzag_homology_persistence(fil,times)
  hosp_graph_auto.append(graph_autocorrelation(G,times,np.linspace(0, T)))
  hosp_dgm.append(dgms)
  plot_diags(dgms,T/20,title = buffer[j],save='figs/'+buffer[j])

auto_array = [top_autocorr(hosp_dgm[i],np.linspace(0, T),format='dionysus') for i in range(7)]
plt.plot(auto_array)
plt.ylim(0,1)
plt.xticks([0,1,2,3,4,5,6],labels=buffer)
plt.ylabel("Topological persistence autocorrelation")
plt.xlabel("Buffer window")
plt.savefig('figs/hosp_top_auto.png')

plt.plot(hosp_graph_auto)
plt.ylim(-1,1)
plt.xticks([0,1,2,3,4,5,6],labels=buffer)
plt.ylabel("Graph autocorrelation")
plt.xlabel("Buffer window")
plt.savefig('figs/hosp_graph_auto.png')

burst = [top_burstiness(dg,T) for dg in hosp_dgm]
plt.plot(np.transpose(burst))
plt.xticks(range(6),labels=["0-birth", "0-death","1-birth","1-death","2-birth","2-death"])
plt.legend(labels=buffer)
plt.title("Burstiness of homological birth and death times")
plt.savefig('figs/hosp_top_burst.png')