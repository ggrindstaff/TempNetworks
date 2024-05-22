import numpy as np 
from teneto import TemporalNetwork
from .utils import teneto_homology
import pandas as pd

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

def scrambly(tempnet):
  #random permutation vector
  n = tempnet.T
  perm = np.array(range(n))
  perm = np.random.permutation(perm)
  edges = np.array(tempnet.network) #columns i j t weight
  newnet = TemporalNetwork()
  for i in range(len(edges)):
    edge = edges[i,:]
    t = edge[2]
    print(t)
    edge[2] = perm[int(t)]
    newnet.add_edge(edge)
  return newnet

def scrambly_arr(tempnet):
  n = tempnet.T
  perm = np.array(range(n))
  perm = np.random.permutation(perm)
  network = tempnet.network
  new_net = np.zeros(shape = network.shape)
  for i in range(n):
    new_net[:,:,i] = network[:,:,perm[i]]
  return TemporalNetwork(from_array = new_net)

def rref_hom(input,m):
    rref = [input]
    results = teneto_homology(input)
    for i in range(1,m):
    #  rref.append(scrambly(input))
        rref.append(scrambly_arr(input))
        results.append(teneto_homology(rref[i]))
    return rref, results

def sun_array(n,T,l):
  sun = np.zeros(shape=(n,n,T))
  for time in range(T):
    t = time%l
    for i in range(n):
      j = i%l
      if j ==1:
        sun[i,(i+1)%n,time] = 1
        sun[i,int((i-l+t)%n),time] = 1
      if 0<j and j < t+1:
        sun[i,(i+1)%n,time] = 1
      if j == t+1:
        sun[i,(i+1)%n,time] = 1
        sun[i,int((i+l-t)%n),time] = 1
      if j > t+1:
        sun[i,(i+1)%n,time] = 1
        sun[i,(i-1)%n,time] = 1
  return sun