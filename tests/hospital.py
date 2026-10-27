#hospital

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempnets as tn
from tempnets.topfunctions import top_autocorr, top_burstiness
from pathlib import Path

current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / 'data'
data_file_path = data_dir / 'detailed_list_of_contacts_Hospital.dat'

hosp = pd.read_csv(data_file_path,delimiter = "\t",header=None)
hosp.columns = ['t','i','j','type1','type2']

"""Contacts are recorded in 20-second steps (t in seconds). We work in hours."""

step = 20/3600
minute = 1/60
hour = 1
day = 24
contacts = [(i,j,t/3600) for (i,j,t) in hosp[['i','j','t']].values.tolist()]
#end of the last recorded step
T = np.max(hosp['t'])/3600 + step

"""We show that different buffer times result in the same topological statistics.

Each contact keeps its edge present for its own 20-second step plus the buffer. A
triangle is present only at the times all three of its pairs are in contact at once,
and is then kept for the same duration."""

buffer_list = [minute,10*minute, hour, 4*hour, 10*hour, day, 2*day]
buffer = ["1 minute","10 minutes", "1 hour", "4 hours", "10 hours", "24 hours", "48 hours"]

hosp_dgm = []
hosp_graph_auto=[]
for j in range(len(buffer_list)):
  duration = step + buffer_list[j]
  net = tn.TemporalNetwork.from_contacts(contacts, duration, T=T)
  cx = tn.TemporalComplex.from_contacts(contacts, duration, max_dim=2, T=T)
  dgms = cx.persistence()
  hosp_graph_auto.append(net.graph_autocorrelation(np.linspace(0, T)))
  hosp_dgm.append(dgms)
  tn.plot_diags(dgms,T,title = buffer[j],save='figs/'+buffer[j])

auto_array = [top_autocorr(hosp_dgm[i],np.linspace(0, T)) for i in range(7)]
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

burst = [top_burstiness(dg,T) for dg in hosp_dgm]
plt.figure()
plt.plot(np.transpose(burst))
plt.xticks(range(6),labels=["0-birth", "0-death","1-birth","1-death","2-birth","2-death"])
plt.legend(labels=buffer)
plt.title("Burstiness of homological birth and death times")
plt.savefig('figs/hosp_top_burst.png')
