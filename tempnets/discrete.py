"""Discrete-time temporal networks, stored as n x n x T arrays:
arr[i,j,t] = 1 when nodes i and j are joined at time step t."""
import numpy as np
from .utils import array_homology

def pairs_to_array(x,n):
  #x[k,t] for the k-th node pair of np.triu_indices(n,1) -> symmetric n x n x T array
  iu = np.triu_indices(n,1)
  arr = np.zeros(shape=(n,n,x.shape[1]))
  arr[iu[0],iu[1],:] = x
  arr[iu[1],iu[0],:] = x
  return arr

def rand_binomial(n,T,birth,stay,init=0):
  #each node pair is a two-state Markov chain: an absent edge appears with probability
  #birth and a present edge stays with probability stay at each step; edges are
  #present at the first step with probability init
  m = n*(n-1)//2
  x = np.zeros(shape=(m,T),dtype=bool)
  x[:,0] = np.random.rand(m) < init
  for t in range(T-1):
    r = np.random.rand(m)
    x[:,t+1] = np.where(x[:,t], r<stay, r<birth)
  return pairs_to_array(x,n)

def rand_poisson(n,T,lam):
  #each node pair has one-step contacts separated by Poisson(lam) gaps; a gap of 0
  #merges two contacts
  m = n*(n-1)//2
  x = np.zeros(shape=(m,T),dtype=bool)
  for k in range(m):
    t = np.random.poisson(lam)
    while t < T:
      x[k,t] = True
      t = t + np.random.poisson(lam)
  return pairs_to_array(x,n)

def to_temp_net(net,times,w):
  #sample a network given by intervals {(u,v): [(start,end),...]} at the window times w
  nodes = list(net.nodes)
  index = {v:i for i,v in enumerate(nodes)}
  n = len(nodes)
  adj = np.zeros(shape=(n,n,len(w)))
  for e in times.keys():
    if len(e)!=2:
      continue
    a = index[e[0]]
    b = index[e[1]]
    for (t1,t2) in times[e]:
      for i in range(len(w)):
        if t1<=w[i]<=t2:
          adj[a,b,i] = 1
          adj[b,a,i] = 1
  return adj

def partition(arr,w,times=None):
  #coarsen time into the windows starting at w (the last window is open-ended): an edge
  #is present in a window if it is present at any of its time steps.
  #times are the labels of the time steps, 0..T-1 by default
  if times is None:
    times = list(range(arr.shape[2]))
  new = np.zeros(shape=(arr.shape[0],arr.shape[1],len(w)))
  for k,time in enumerate(times):
    for i in range(len(w)):
      if (i == len(w)-1 and time>=w[i]) or (i < len(w)-1 and w[i]<=time<w[i+1]):
        new[:,:,i] = np.maximum(new[:,:,i],arr[:,:,k])
  return new

def scrambly_arr(arr):
  #randomly permute the time steps
  perm = np.random.permutation(arr.shape[2])
  return arr[:,:,perm]

def rref_hom(input,m):
    rref = [input]
    results = [array_homology(input)]
    for i in range(1,m):
        rref.append(scrambly_arr(input))
        results.append(array_homology(rref[i]))
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