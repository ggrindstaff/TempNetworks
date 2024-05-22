import teneto
from teneto import TemporalNetwork

import dionysus as d
import scipy.spatial.distance as distance
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import itertools
import pandas as pd
import random
#import persim
np.random.seed(10)

from matplotlib import rc
rc('animation', html='jshtml')

"""#utils

Level set function for temporal graph, filters to the subgraph present at time t. This takes input/output ordered list of times, not dictionary (see other notebook for that)
"""

def level_set(G,times,t):
  H = nx.Graph()
  H.add_nodes_from(G)
  n = len(G.nodes)
  edges = list(G.edges)
  on = True
  for i in range(len(edges)):
    ints = times[i+n]
    for j in range(len(ints)):
      if j==len(ints)-1:
        if on and t>ints[j]:
          edge = edges[i]
          H.add_edge(edge[0],edge[1])
      if on and ints[j]<t and ints[j+1]>t:
        edge = edges[i]
        H.add_edge(edge[0],edge[1])
        break
      on = not on
  return H

"""Turn X (nodes and edges already added) into a temporal flag complex

The ```intersection``` function computes the intersection of n lists of times, for a temporal simplicial flag complex.
"""

def intersection(all_times):
  if len(all_times)==1:
    return all_times[0]
  list1=all_times[0]
  list2=all_times[1]
  inter_times=[]
  for (t1,t2) in list1:
    for (s1,s2) in list2:
      (u,v) = (t1,t2)
      if u>=s2 or v<=s1:
        inter_bool=False
      else:
        inter_bool=True
        if u<s1:
          u=s1
        if v>s2:
          v=s2
      if inter_bool:
        inter_times.append((u,v))
  if inter_times==[]:
    return []
  new = all_times[2:]
  return intersection(all_times[2:] + [inter_times])

"""## Time list functions

The dionysus version needs to be converted into the (simplices, times) format. We do this first for edges, adding, then convert to flag zigzag complex.
"""

#bank is a sorted list of integer timestamps that a particular simplex is present
# if timestamps are non-integer, we would require a list of all measured times,
#   to be addressed in separate function
def time_list_to_ee(b,T,inter='union',step=1):
  bank = b.copy()
  if inter=='union':
    alpha = 0.5
    beta = 0.5
  if inter=='intersection':
    alpha = 0.4
    beta= 0
  if inter=='flag':
    alpha=0
    beta=0
  time = bank.pop(0)
  times = [max(time-beta,0)]
  while len(bank)>0:
    if bank[0] == time +step:
      time = bank.pop(0)
    else:
      times.append(time + alpha)
      times.append(bank[0]-beta)
      time = bank.pop(0)
  if time < T:
    times.append(time+alpha)
  return times

def timelist_to_ee_non_int(timelist, all_times,buffer=1,**kwargs):
  buffer = buffer+1
  new_times = [all_times.index(t) for t in timelist]
  ntimes_buf=[]
  for i in range(len(new_times)-1):
    t = new_times[i]
    next_t = new_times[i+1]
    if t + buffer >= next_t:
      ntimes_buf = ntimes_buf + list(range(t,next_t,1))
    else:
      ntimes_buf = ntimes_buf + list(range(t,t+buffer,1))
  i = new_times[-1]
  if i+buffer>=len(all_times):
    ntimes_buf = ntimes_buf + list(range(i,len(all_times),1))
  else:
    ntimes_buf = ntimes_buf + list(range(i,i+buffer,1))
  ee = time_list_to_ee(ntimes_buf,len(all_times),**kwargs)
  return ee

def to_ee(time_dict,sxs,T):
  dtimes=[]
  for sx in sxs:
    if len(sx)==1:
      dtimes.append([0,T])
    else:
      dtimes.append(list(itertools.chain.from_iterable(time_dict[tuple(sx)])))
  return dtimes

def ee_to_time_list(times,T,buffer=0.5):
  bank = []
  time =0
  for i in range(len(times)):
    time = times[i]
    if time>=T:
      break
    if i%2==0:
      bank.append(time)
      while time<=times[i+1]-buffer and time <=T:
        time = time+buffer
        bank.append(time)
  return bank

def is_in(time,t):
  flag = False
  for (t1,t2) in time:
    if t1<=t<=t2:
      flag=True
  return flag

def is_in_ee(dtimes,t):
  for i in range(len(dtimes)-1):
    if dtimes[i]<=t<=dtimes[i+1]:
      return i%2==0
  return False

"""## Simplicial functions

This function checks whether a zigzag complex (represented by a list of simplices sxs and a corresponding dictionary of timesteps timebank) is in fact a simplicial complex at each timestep.
"""

def check_simplicial(sxs,timebank):
  for simplex in sxs:
    if len(simplex)<=2:
      continue
    else:
      sx = tuple(simplex)
      times_to_check = timebank[sx]
      boundary = list(itertools.combinations(sx,len(sx)-1))
      if all([list(b) in sxs for b in boundary]):
        for t in times_to_check:
          if all([t in timebank[b] for b in boundary]):
            continue
          else: return False
      else: return False
  return True

"""converts between Temporal Network object and zigzag simplical complex"""
"""
def temp_net_to_clique_clx(temp_net,inter='union',max_dim=2):
  T = temp_net.T
  n = temp_net.N
  simplices = [[i] for i in range(n)]
  time_bank = {}
  times = [[0,T] for i in range(n)]
  for i in range(n):
    #time_bank[i] = [j for j in range(T)]
    for j in range(i):
      ij = temp_net.get_network_when(i=i,j=j)
      ji = temp_net.get_network_when(i=j,j=i)
      if (ij.size+ji.size)!=0:
        simplices.append([j,i])
        bank = sorted(list(ij['t'].values) + [x for x in (ji['t'].values) if x not in list(ij['t'].values)])
        ee = time_list_to_ee(bank,T,inter)
        times.append(ee)
      #time_bank[(j,i)] = bank

  for i in range(len(simplices)):
    time_bank[tuple(simplices[i])] = ee_to_time_list(times[i],T)
  #add higher simplices to make clique complex
  for i in range(2,max_dim+1):
    possible = list(itertools.combinations(range(n),i+1))
    for sx in possible:
      boundary1 = list(itertools.combinations(sx,2))
      if all([list(b) in simplices for b in boundary1]):
        time = set(time_bank[boundary1[0]])
        for b in boundary1:
          time =  set.intersection(time,set(time_bank[b]))
        if len(time)>0:
          simplices.append(list(sx))
          time_bank[sx] = list(set(sorted(time)))
          times.append(time_list_to_ee(time_bank[sx],T,'flag',step=0.5))
  #sanity check to make sure all faces are added properly
  if check_simplicial(simplices,time_bank)==False:
    raise IndexError
  D = d.Filtration(simplices)
  return simplices, times,time_bank
  #d.zigzag_homology_persistence(D, times)
"""
def temp_net_to_clique_clx(temp_net,inter='union',max_dim=2):
  T = temp_net.T
  n = temp_net.N
  simplices = [[i] for i in range(n)]
  time_bank = {}
  times = [[0,T] for i in range(n)]
  for i in range(n):
    #time_bank[i] = [j for j in range(T)]
    for j in range(i):
      ij = temp_net.get_network_when(i=i,j=j)
      ji = temp_net.get_network_when(i=j,j=i)
      if (ij.size+ji.size)!=0:
        simplices.append([j,i])
        bank = sorted(list(ij['t'].values) + [x for x in (ji['t'].values) if x not in list(ij['t'].values)])
        ee = time_list_to_ee(bank,T,inter)
        times.append(ee)
      #time_bank[(j,i)] = bank

  for i in range(len(simplices)):
    time_bank[tuple(simplices[i])] = ee_to_time_list(times[i],T)
  #add higher simplices to make clique complex
  for i in range(2,max_dim+1):
    possible = list(itertools.combinations(range(n),i+1))
    for sx in possible:
      boundary1 = list(itertools.combinations(sx,2))
      if all([list(b) in simplices for b in boundary1]):
        time = set(time_bank[boundary1[0]])
        for b in boundary1:
          time =  set.intersection(time,set(time_bank[b]))
        if len(time)>0:
          simplices.append(list(sx))
          time_bank[sx] = list(set(sorted(time)))
          times.append(time_list_to_ee(time_bank[sx],T,'flag',step=0.5))
  #sanity check to make sure all faces are added properly
  if check_simplicial(simplices,time_bank)==False:
    raise IndexError
  D = d.Filtration(simplices)
  return simplices, times
  #d.zigzag_homology_persistence(D, times)

"""## Random model generators"""

def periodic_intervals(T,freq,l,tda = 'bats'):
  start = np.random.rand()*freq
  dur = np.random.exponential(l)
  t = [(start,start+dur)]
  while True:
    newstart = t[-1][1] + freq + np.random.normal(0,1)
    if newstart >= T: break
#    dur = np.random.exponential(l)
    dur = l
    if newstart + dur <T:
      t.append((newstart,newstart + dur))
#      print((newstart,newstart+dur))
    else:
      t.append((newstart,T))
      break
  if tda=='dionysus':
    return list(itertools.chain.from_iterable(t))
  return t

def ER_periodic(n,p,freq,l,T,max_dim):
  net = nx.Graph(directed=False)
  sxs = []
  #vertices remain in the network for the entire interval
  for i in range(n):
    sxs.append([i])
    net.add_node(i)

  times = {}
  #add in edges via Erdos-Renyi uniform random
  for i in range(n):
    for j in range(i):
      roll = np.random.rand()
      if roll < p:
        net.add_edge(j,i)
        sxs.append([j,i])
        t = periodic_intervals(T,freq,l)
        if t!=[]:
          times[(j,i)] = t

  for i in range(2,max_dim+1):
    possible = list(itertools.combinations(range(n),i+1))
    for sx in possible:
      boundary1 = list(itertools.combinations(sx,2))
      if all([list(b) in sxs for b in boundary1]):
        time = intersection([times[b] for b in boundary1])
        if len(time)>0:
          sxs.append(list(sx))
          times[sx] = time

  return net,sxs,times

def exponential_rewiring(net,l,T,max_dim=2):
    n = net.number_of_nodes()
    sxs = []
    for i in range(n):
      sxs.append([i])
    times = {}
    for edge in net.edges():
      times[edge] = [(0,T)]
    t = int(np.random.exponential(l)*1000)/1000
    num = len(net.edges())
    G = net.copy()
    while t < T:
      lineup = list(G.edges())
      togo = lineup[np.random.randint(num)]
      G.remove_edge(togo[0],togo[1])
      recent = times[togo][-1]
      times[togo][-1] = (recent[0],t)
      stay = togo[np.random.randint(2)]
      new_neighbor = np.random.randint(n)
      while new_neighbor == stay or (stay,new_neighbor) in G.edges() or (new_neighbor,stay) in G.edges() or (stay,new_neighbor)==togo or (new_neighbor,stay)==togo:
        new_neighbor = np.random.randint(n)
      G.add_edge(stay,new_neighbor)
      u = min(stay,new_neighbor)
      v = max(stay,new_neighbor)
      if (u,v) in times.keys():
        if times[(u,v)][-1][1]==T:
          print(str((u,v)) + "bad bad no good")
        times[(u,v)].append((t,T))
      else:
        times[(u,v)] =[(t,T)]
      t = int((t + np.random.exponential(l))*1000)/1000
    for edge in times.keys():
      sxs.append(list(edge))
      G.add_edge(edge[0],edge[1])
#      for (t0,t1) in times[edge]:
#        X.add(t0,t1,[edge[0],edge[1]])
#    X = flag(X,times,max_dim=max_dim)

    for i in range(2,max_dim+1):
      possible = list(itertools.combinations(range(n),i+1))
      for sx in possible:
        boundary1 = list(itertools.combinations(sx,2))
        if all([list(b) in sxs for b in boundary1]):
          time = intersection([times[b] for b in boundary1])
          if len(time)>0:
            sxs.append(list(sx))
            times[sx] = time

    return G,sxs,times

"""## Plotting and visualization"""

def plot_diags(dgms,T,rotate=False,title="",save=""):
  eps = 0.1*T
  birth = []
  death = []
  pers = []
  infinity_line = T+eps
  for i,dgm in enumerate(dgms):
    newbirth = []
    newdeath = []
    newpers = []
    #print("Dimension",i)
    for p in dgm:
      newbirth.append(p.birth)
      if p.death < T:
        newdeath.append(p.death)
        newpers.append(p.death-p.birth)
      else:
        newdeath.append(infinity_line)
        newpers.append(infinity_line)
    birth.append(newbirth)
    death.append(newdeath)
    pers.append(newpers)

  colors = ['blue','orange','purple','red','gold','lime']
  plotmin = np.min([np.min(birth[i]) for i in range(len(birth))])
  plotmax= np.max([np.max(death[i]) for i in range(len(death))])
  #rot_plotmax = np.max([np.max(all_finite_pers[i]) for i in range(len(all_finite_pers))])

  if title=="":
    title = "Zigzag Persistence Diagram"

  if rotate:
    for dim in range(len(birth)):
      plt.scatter(birth[dim],[pers[dim][i] for i in range(len(birth[dim]))],color = colors[dim])
    #plt.scatter(birth[0],[death[0][i]-birth[0][i] for i in range(len(birth[0]))],color='blue')
    #plt.scatter(birth[1],[death[1][i]-birth[1][i] for i in range(len(birth[1]))],color='orange')
    plt.xlim(plotmin-eps,plotmax+eps)
    plt.ylim(0,infinity_line+1)
    plt.text(plotmin-2*eps,infinity_line, "$\infty$",size=15)
    plt.plot([plotmin-eps,plotmax+eps], [infinity_line,infinity_line], '--', linewidth=2, markersize=0)
    plt.title(title)
    plt.xlabel("Birth time")
    plt.ylabel("Persistence")
    if save=="":
      plt.show()
    else:
      plt.savefig(save+'.png')
      plt.show()

  else:
    plt.plot([plotmin,T+eps], [plotmin,T+eps], 'go-', linewidth=2, markersize=0)
    plt.xlim(plotmin,T+eps)
    plt.ylim(plotmin,infinity_line + eps)
    plt.text(-0.1*T,infinity_line, "$\infty$",size=15)
    plt.plot([plotmin,plotmax+eps], [infinity_line,infinity_line], '--', linewidth=2, markersize=0)
    for i in range(len(birth)):
      plt.scatter(birth[i],death[i],color=colors[i],s=10*(i+1),alpha = 0.8,label=i)
    plt.yticks(np.linspace(0, T, 8))
    plt.legend(loc='lower right')
    plt.title(title)
    plt.xlabel("Birth time")
    plt.ylabel("Death time")
    if save=="":
      plt.show()
    else:
      plt.savefig(save+'.png')
      plt.show()

"""Might need to rewrite this one, it is a duplicate

## Network Windowing

This will convert a network with list of entry and exit times into a discrete temporal network via the window set w
"""

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

def teneto_homology(input):
    sxs, times = temp_net_to_clique_clx(input)
    #simplices = [sxs]
    #eelist = [times]
    # flag function
    zz,dgms,cell= d.zigzag_homology_persistence(d.Filtration(sxs),times)
    results = [dgms]
    return dgms

def remove_a_loop(G):
  try:
    edges = nx.simple_cycles(G)
    edge = np.random.randint(0,len(edges))
    (s,t) = edges[edge]
    G.remove_edge(s,t)
    return (s,t)
  except:
    return None

def add_rand_edge(G):
  n = G.number_of_nodes()
  u = np.random.randint(n)
  v = np.random.randint(n)
  while u==v:
    v=np.random.randint(n)
  if G.has_edge(u,v) or G.has_edge(v,u):
    return add_rand_edge(G)
  else:
    G.add_edge(min(u,v),max(u,v))
    return (min(u,v),max(u,v))

def remove_rand_edge(G):
  to_remove = random.sample(G.edges(),k=1)
  G.remove_edges_from(to_remove)
  return to_remove[0]

#This is the basic building function

def topological_rewiring(net,l,T,max_dim=2):
    sxs = []
    n = net.N
    for i in range(n):
      sxs.append([i])
    times = {}
    for edge in net.edges():
      times[edge] = [(0,T)]
    t = int(np.random.exponential(l)*1000)/1000
    num = len(net.edges())
    G = net.copy()
    while t < T:
      edge = remove_a_loop(G)
      if edge==None:
        (a,b) = remove_rand_edge(G)
      else:
        a = edge[0]
        b = edge[1]
      recent = times[(a,b)][-1]
      times[(a,b)][-1] = (recent[0],t)

      (u,v) = add_rand_edge(G)
      if (u,v) in times.keys():
        if times[(u,v)][-1][1]==T:
          print(str((u,v)) + "bad bad no good")
        times[(u,v)].append((t,T))
      else:
        times[(u,v)] =[(t,T)]
      t = int((t + np.random.exponential(l))*1000)/1000
    for edge in times.keys():
      sxs.append(list(edge))
      G.add_edge(edge[0],edge[1])
#      for (t0,t1) in times[edge]:
#        X.add(t0,t1,[edge[0],edge[1]])
#    X = flag(X,times,max_dim=max_dim)

    for i in range(2,max_dim+1):
      possible = list(itertools.combinations(range(n),i+1))
      for sx in possible:
        boundary1 = list(itertools.combinations(sx,2))
        if all([list(b) in sxs for b in boundary1]):
          time = intersection([times[b] for b in boundary1])
          if len(time)>0:
            sxs.append(list(sx))
            times[sx] = time

    return G,sxs,times
