"""Parameters for the random models, computed from target statistics.

Every model is specified by
  T     the number of time steps
  dens  the average density: the expected fraction of the n(n-1)/2 node pairs
        that are joined by an edge at any given time
  rate  the average rate of new edge generation: the expected number of edge
        appearances (absent -> present) per node pair per unit time, i.e.
        rate*n(n-1)/2 new edges per time step across the whole network

In a stationary network these fix the mean lifetime of an edge at dens/rate
(Little's law), whatever the model.
"""
from math import comb, exp
import numpy as np
from scipy.optimize import brentq


def edge_lifetime(dens,rate):
  return dens/rate

def binomial_params(dens,rate):
  """prob=[P(absent -> present), P(present stays present)] for teneto's rand_binomial.

  Each node pair is a two-state Markov chain; at stationarity
  P(present) = dens and P(absent)*P(absent -> present) = rate.
  Pass initialize=dens as well so the network starts at stationarity.
  """
  birth = rate/(1-dens)
  death = rate/dens
  if birth > 1 or death > 1:
    raise ValueError("rand_binomial needs rate <= min(dens, 1-dens)")
  return [birth, 1-death]

def poisson_params(dens,T):
  """(ncontacts, lam) for teneto's rand_poisson.

  Contacts last one time step and the gaps between them are Poisson(lam);
  zero gaps merge contacts, so the density is (1-exp(-lam))/lam. This model
  has no separate duration parameter, so its rate of new edges cannot be set
  independently of dens.

  rand_poisson makes the network as long as the longest pair's contact
  sequence, leaving the other pairs empty at the end; ncontacts is large
  enough that every pair lasts past T, so crop the result to its first T steps.
  """
  lam = brentq(lambda x: (1-exp(-x))/x - dens, 1e-9, 1e6)
  ncontacts = int(np.ceil((T + 6*np.sqrt(T) + 6)/lam))
  return ncontacts, lam

def er_periodic_params(dens,rate,p):
  """(freq, l) for ER_periodic with edge probability p.

  Each edge of a G(n,p) graph is present for l time steps out of every
  freq + l, so dens = p*l/(freq+l) and rate = p/(freq+l). p is a free
  choice and must satisfy dens < p <= 1.
  """
  if not dens < p <= 1:
    raise ValueError("ER_periodic needs dens < p <= 1")
  l = edge_lifetime(dens,rate)
  period = p/rate
  return period - l, l

def rewiring_interval(n,rate):
  """Mean time between rewiring events (one new edge each) for exponential_rewiring
  and topological_rewiring."""
  return 1/(rate*comb(n,2))

def ws_params(n,dens,rate):
  """(k, l) for a Watts-Strogatz graph rewired by exponential_rewiring.

  Rewiring keeps the number of edges n*k/2 fixed, so dens = k/(n-1); k is
  rounded to the nearest even number, so the density is matched as closely
  as the graph allows.
  """
  k = max(2, 2*int(round(dens*(n-1)/2)))
  return k, rewiring_interval(n,rate)

def interval_stats(times,n,T):
  """(density, rate) of an interval network {(u,v): [(start,end),...]} on [0,T]."""
  pairs = comb(n,2)
  on = 0
  births = 0
  for e in times.keys():
    if len(e)!=2:
      continue
    for (s,t) in times[e]:
      on = on + max(0, min(t,T) - max(s,0))
      if 0 < s < T:
        births = births + 1
  return on/(pairs*T), births/(pairs*T)

def array_stats(arr):
  """(density, rate) of an undirected binary n x n x T array."""
  n = arr.shape[0]
  iu = np.triu_indices(n,1)
  x = arr[iu[0],iu[1],:] > 0
  births = np.logical_and(~x[:,:-1], x[:,1:]).sum()
  return x.mean(), births/(x.shape[0]*(x.shape[1]-1))
