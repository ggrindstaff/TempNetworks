"""Random temporal network models, specified by target statistics.

Every model takes
  n        number of nodes
  T        length of the time interval [0, T]
  density  average fraction of the n(n-1)/2 node pairs joined by an edge at a time
  rate     average number of new edges per node pair per unit time
and returns a TemporalNetwork; net.stats() reports what was achieved. Edges last
density/rate on average. Pass seed for a reproducible network (numpy's global
random state is left as it was).
"""
from contextlib import contextmanager

import networkx as nx
import numpy as np

from . import params
from .discrete import rand_binomial, rand_poisson, sun_array
from .network import TemporalNetwork
from .utils import ER_periodic, exponential_rewiring, topological_rewiring as _topological_rewiring


@contextmanager
def seeded(seed):
  #run with np.random seeded, restoring the global random state afterwards
  if seed is None:
    yield
    return
  state = np.random.get_state()
  np.random.seed(seed)
  try:
    yield
  finally:
    np.random.set_state(state)

def edge_intervals(times):
  #the edges of an interval dict that may also hold higher simplices
  return {e: ints for e, ints in times.items() if len(e) == 2}


def binomial(n, T, density, rate, seed=None):
  """Discrete time: each node pair is a two-state Markov chain, started at stationarity.
  Needs rate <= min(density, 1 - density)."""
  birth, stay = params.binomial_params(density, rate)
  with seeded(seed):
    return TemporalNetwork.from_array(rand_binomial(n, T, birth, stay, init=density))

def poisson(n, T, density, seed=None):
  """Discrete time: one-step contacts separated by Poisson gaps. Contacts last a single
  step, so the rate of new edges follows from the density and cannot be chosen."""
  with seeded(seed):
    return TemporalNetwork.from_array(rand_poisson(n, T, params.poisson_params(density)))

def er_periodic(n, T, density, rate, p=None, seed=None):
  """Each edge of a G(n, p) graph switches on for density/rate time units, roughly
  periodically. p must satisfy density < p <= 1; by default p = min(1, 2*density),
  so edges are present about half the time."""
  if p is None:
    p = min(1.0, 2 * density)
  freq, l = params.er_periodic_params(density, rate, p)
  with seeded(seed):
    net, sxs, times = ER_periodic(n, p, freq, l, T, 1)
  return TemporalNetwork.from_intervals(edge_intervals(times), T, nodes=range(n))

def ws_rewiring(n, T, density, rate, seed=None):
  """A ring lattice (Watts-Strogatz with no shortcuts) whose edges are rewired one at a
  time at exponential intervals. The number of edges is fixed, so the density is
  matched as closely as an even neighbour count allows."""
  k, l = params.ws_params(n, density, rate)
  G = nx.watts_strogatz_graph(n, k, 0)
  with seeded(seed):
    net, sxs, times = exponential_rewiring(G, l, T, 1)
  return TemporalNetwork.from_intervals(edge_intervals(times), T, nodes=range(n))

def topological_rewiring(n, T, density, rate, seed=None):
  """Topologically self-inhibiting: starting from G(n, density), each rewiring removes an
  edge lying on a cycle (when there is one) and adds a random new edge."""
  with seeded(seed):
    G = nx.gnp_random_graph(n, density, seed=np.random.randint(2**31))
    net, sxs, times = _topological_rewiring(G, params.rewiring_interval(n, rate), T, 1)
  return TemporalNetwork.from_intervals(edge_intervals(times), T, nodes=range(n))

def sun(n, T, rays=5):
  """A deterministic discrete-time network of rays (see discrete.sun_array); not matched
  to a density or rate."""
  return TemporalNetwork.from_array(sun_array(n, T, n / rays))
