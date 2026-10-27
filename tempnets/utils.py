import dionysus as d
import scipy.spatial.distance as distance
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import itertools
from bisect import bisect_right
#import persim

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
  for i in range(len(edges)):
    #times[i+n] is the entry/exit list of the i-th edge of G.edges; t is inside an
    #on-interval when an odd number of entries/exits are <= t
    if bisect_right(times[i+n],t)%2==1:
      edge = edges[i]
      H.add_edge(edge[0],edge[1])
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
  if inter not in ('union','intersection','flag'):
    raise ValueError("inter must be 'union', 'intersection' or 'flag'")
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
  index = {t:i for i,t in enumerate(all_times)}
  new_times = [index[t] for t in timelist]
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

def graph_ee(G,time_dict,T):
  #entry/exit lists in the order level_set expects: nodes, then the edges of G.edges
  dtimes = [[0,T] for v in G.nodes]
  for (u,v) in G.edges:
    dtimes.append(list(itertools.chain.from_iterable(time_dict[(min(u,v),max(u,v))])))
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

def as_arrays(dgms):
  #persistence diagrams in any of the package's formats (dionysus diagrams, lists of
  #(birth, death) tuples, or arrays) -> one float array of (birth, death) rows per dimension
  out = []
  for dgm in dgms:
    if isinstance(dgm,np.ndarray):
      out.append(dgm.astype(float).reshape(-1,2))
    else:
      out.append(np.array([(p.birth,p.death) if hasattr(p,'birth') else tuple(p) for p in dgm],dtype=float).reshape(-1,2))
  return out

"""## Zigzag persistence

dionysus stores times as 32-bit floats (resolution about 4e-6 at t=50), so an interval
shorter than that collapses to a point and dionysus rejects the whole filtration.
"""

def snap_ee(ee):
  #round an entry/exit list to float32, dropping intervals that collapse to a point and
  #merging intervals whose gap collapses. Rounding is monotone, so a coface's
  #intervals stay inside its faces' intervals.
  out = []
  for x in np.asarray(ee,dtype=np.float32).tolist():
    if len(out)>0 and x<=out[-1]:
      out.pop()
    else:
      out.append(x)
  return out

def zigzag_persistence(sxs,dtimes):
  #d.zigzag_homology_persistence on a list of simplices (or a d.Filtration) and their
  #entry/exit lists, with times made safe for dionysus; returns (zz, dgms, cells)
  f = sxs if isinstance(sxs,d.Filtration) else d.Filtration(sxs)
  return d.zigzag_homology_persistence(f,[snap_ee(ee) for ee in dtimes])

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

"""converts between a discrete temporal network (n x n x T array) and zigzag simplical complex"""
def temp_net_to_clique_clx(arr,inter='union',max_dim=2):
  #arr is an n x n x T array (see discrete.py); arr[i,j,t] or arr[j,i,t] nonzero
  #means i and j are joined at time step t
  n = arr.shape[0]
  T = arr.shape[2]
  simplices = [[i] for i in range(n)]
  intervals = {}
  times = [[0,T] for i in range(n)]
  for i in range(n):
    for j in range(i):
      bank = [int(t) for t in np.nonzero((arr[i,j,:]!=0) | (arr[j,i,:]!=0))[0]]
      if len(bank)>0:
        simplices.append([j,i])
        ee = time_list_to_ee(bank,T,inter)
        times.append(ee)
        if len(ee)%2==1:
          ee = ee + [T]
        intervals[(j,i)] = list(zip(ee[::2],ee[1::2]))

  #add higher simplices to make clique complex; a clique is present on the
  #(open) intervals where all of its edges are present
  edges = set(intervals.keys())
  for i in range(2,max_dim+1):
    possible = list(itertools.combinations(range(n),i+1))
    for sx in possible:
      boundary1 = list(itertools.combinations(sx,2))
      if all([b in edges for b in boundary1]):
        time = intersection([intervals[b] for b in boundary1])
        if len(time)>0:
          simplices.append(list(sx))
          times.append(list(itertools.chain.from_iterable(time)))
  return simplices, times

"""## Random model generators"""

def periodic_intervals(T,freq,l,tda = 'bats'):
  if freq <= 0:
    raise ValueError("freq (gap between appearances) must be positive")
  start = np.random.rand()*freq
  dur = np.random.exponential(l)
  t = [(start,start+dur)]
  while True:
    #jitter the gap, but never let consecutive intervals overlap or touch
    newstart = t[-1][1] + max(freq + np.random.normal(0,1), 1e-3)
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
    t = np.random.exponential(l)
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
      t = t + np.random.exponential(l)
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
        if all([b in times for b in boundary1]):
          time = intersection([times[b] for b in boundary1])
          if len(time)>0:
            sxs.append(list(sx))
            times[sx] = time

    return G,sxs,times

"""## Plotting and visualization"""

def plot_diags(dgms,T,rotate=False,title="",save=""):
  fig = plt.figure()
  eps = 0.1*T
  birth = []
  death = []
  pers = []
  infinity_line = T+eps
  for i,dgm in enumerate(as_arrays(dgms)):
    newbirth = []
    newdeath = []
    newpers = []
    #print("Dimension",i)
    for (b,dth) in dgm:
      newbirth.append(b)
      if dth < T:
        newdeath.append(dth)
        newpers.append(dth-b)
      else:
        newdeath.append(infinity_line)
        newpers.append(infinity_line)
    birth.append(newbirth)
    death.append(newdeath)
    pers.append(newpers)

  colors = ['blue','orange','purple','red','gold','lime']
  #diagrams of some dimensions can be empty
  plotmin = min([min(b) for b in birth if len(b)>0], default=0)
  plotmax = max([max(dth) for dth in death if len(dth)>0], default=T)
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
    plt.text(plotmin-2*eps,infinity_line, r"$\infty$",size=15)
    plt.plot([plotmin-eps,plotmax+eps], [infinity_line,infinity_line], '--', linewidth=2, markersize=0)
    plt.title(title)
    plt.xlabel("Birth time")
    plt.ylabel("Persistence")
    if save!="":
      plt.savefig(save+'.png')
    plt.show()
    plt.close(fig)

  else:
    plt.plot([plotmin,T+eps], [plotmin,T+eps], 'go-', linewidth=2, markersize=0)
    plt.xlim(plotmin,T+eps)
    plt.ylim(plotmin,infinity_line + eps)
    plt.text(-0.1*T,infinity_line, r"$\infty$",size=15)
    plt.plot([plotmin,plotmax+eps], [infinity_line,infinity_line], '--', linewidth=2, markersize=0)
    for i in range(len(birth)):
      plt.scatter(birth[i],death[i],color=colors[i],s=10*(i+1),alpha = 0.8,label=i)
    plt.yticks(np.linspace(0, T, 8))
    plt.legend(loc='lower right')
    plt.title(title)
    plt.xlabel("Birth time")
    plt.ylabel("Death time")
    if save!="":
      plt.savefig(save+'.png')
    plt.show()
    plt.close(fig)

def array_homology(input):
    sxs, times = temp_net_to_clique_clx(input)
    #simplices = [sxs]
    #eelist = [times]
    # flag function
    zz,dgms,cell= zigzag_persistence(sxs,times)
    results = [dgms]
    return dgms

def remove_a_loop(G):
  #remove a random edge lying on a cycle; None if G is a forest
  cycles = nx.cycle_basis(G)
  if len(cycles)==0:
    return None
  cycle = cycles[np.random.randint(len(cycles))]
  i = np.random.randint(len(cycle))
  (s,t) = (cycle[i],cycle[(i+1)%len(cycle)])
  G.remove_edge(s,t)
  return (min(s,t),max(s,t))

def add_rand_edge(G,exclude=None):
  #add a uniformly random absent edge, never the edge `exclude`
  n = G.number_of_nodes()
  while True:
    u = np.random.randint(n)
    v = np.random.randint(n)
    e = (min(u,v),max(u,v))
    if u!=v and not G.has_edge(u,v) and e!=exclude:
      G.add_edge(e[0],e[1])
      return e

def remove_rand_edge(G):
  edges = list(G.edges())
  (u,v) = edges[np.random.randint(len(edges))]
  G.remove_edge(u,v)
  return (min(u,v),max(u,v))

#This is the basic building function

def topological_rewiring(net,l,T,max_dim=2):
    sxs = []
    n = net.number_of_nodes()
    for i in range(n):
      sxs.append([i])
    times = {}
    for edge in net.edges():
      times[(min(edge),max(edge))] = [(0,T)]
    t = np.random.exponential(l)
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

      (u,v) = add_rand_edge(G,exclude=(a,b))
      if (u,v) in times.keys():
        if times[(u,v)][-1][1]==T:
          print(str((u,v)) + "bad bad no good")
        times[(u,v)].append((t,T))
      else:
        times[(u,v)] =[(t,T)]
      t = t + np.random.exponential(l)
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
        if all([b in times for b in boundary1]):
          time = intersection([times[b] for b in boundary1])
          if len(time)>0:
            sxs.append(list(sx))
            times[sx] = time

    return G,sxs,times
