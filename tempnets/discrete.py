"""Discrete-time temporal networks, stored as n x n x T arrays:
arr[i,j,t] = 1 when nodes i and j are joined at time step t."""
import numpy as np
from .utils import array_homology

def pairs_to_array(x,n):
  """Turn x[k, t], indexed by the k-th node pair of np.triu_indices(n, 1), into a
  symmetric n x n x T array."""
  iu = np.triu_indices(n,1)
  arr = np.zeros(shape=(n,n,x.shape[1]))
  arr[iu[0],iu[1],:] = x
  arr[iu[1],iu[0],:] = x
  return arr

def rand_binomial(n,T,birth,stay,init=0):
  """Discrete-time network where each node pair is a two-state Markov chain: an
  absent edge appears with probability birth and a present edge stays with
  probability stay at each step; edges are present at the first step with
  probability init. Returns an n x n x T array. See generators.binomial."""
  m = n*(n-1)//2
  x = np.zeros(shape=(m,T),dtype=bool)
  x[:,0] = np.random.rand(m) < init
  for t in range(T-1):
    r = np.random.rand(m)
    x[:,t+1] = np.where(x[:,t], r<stay, r<birth)
  return pairs_to_array(x,n)

def rand_poisson(n,T,lam):
  """Discrete-time network where each node pair has one-step contacts separated
  by Poisson(lam) gaps (a gap of 0 merges two contacts), started at stationarity:
  the gaps between distinct contacts are Poisson(lam) conditioned to be >= 1, and
  the first contact time is drawn from the stationary forward-recurrence
  distribution P(first = r) = P(gap > r)/E[gap]. Returns an n x n x T array.
  See generators.poisson."""
  from scipy.stats import poisson
  m = n*(n-1)//2
  #gap between distinct contacts: Poisson(lam) given >= 1
  rmax = int(lam + 12*np.sqrt(lam) + 12)
  g = np.arange(1, rmax+1)
  pgap = poisson.pmf(g, lam)/(1 - np.exp(-lam))
  tail = np.clip(1 - np.concatenate([[0], np.cumsum(pgap)])[:rmax], 0, None)
  first = tail/tail.sum()
  x = np.zeros(shape=(m,T),dtype=bool)
  for k in range(m):
    t = np.random.choice(rmax, p=first)
    while t < T:
      x[k,t] = True
      gap = 0
      while gap == 0:
        gap = np.random.poisson(lam)
      t = t + gap
  return pairs_to_array(x,n)

def to_temp_net(net,times,w):
  """Sample a network given by intervals {(u, v): [(start, end), ...]} at the
  times w: an n x n x len(w) array with nodes in the order of net.nodes. See also
  TemporalNetwork.to_array."""
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
  """Coarsen an n x n x T array into the windows starting at the times w (the last
  window is open-ended): an edge is present in a window if it is present at any of
  its steps. times labels the steps (0, ..., T-1 by default)."""
  if times is None:
    times = list(range(arr.shape[2]))
  new = np.zeros(shape=(arr.shape[0],arr.shape[1],len(w)))
  for k,time in enumerate(times):
    for i in range(len(w)):
      if (i == len(w)-1 and time>=w[i]) or (i < len(w)-1 and w[i]<=time<w[i+1]):
        new[:,:,i] = np.maximum(new[:,:,i],arr[:,:,k])
  return new

def scrambly_arr(arr):
  """The array with its time steps randomly permuted (a null model that keeps
  each snapshot but destroys temporal order)."""
  perm = np.random.permutation(arr.shape[2])
  return arr[:,:,perm]

def rref_hom(input,m):
    """The array input and m - 1 time-permuted copies (scrambly_arr), with the zigzag
    diagrams of each (array_homology). Returns (arrays, diagrams)."""
    rref = [input]
    results = [array_homology(input)]
    for i in range(1,m):
        rref.append(scrambly_arr(input))
        results.append(array_homology(rref[i]))
    return rref, results

def sun_array(n,T,l):
  """Deterministic n x n x T test network: the n nodes are arranged in rays of l
  nodes along a ring, and connections between and within rays shift with period
  l. Used as a structured example in models.py; see generators.sun."""
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