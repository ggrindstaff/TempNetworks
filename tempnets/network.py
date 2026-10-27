"""Temporal networks and their temporal clique complexes.

A TemporalNetwork lives on the time interval [0, T]. Its nodes are present the
whole time and each edge is present on a list of disjoint intervals. This one
object replaces the separate formats (interval dicts, entry/exit lists in
simplex order or in G.edges order, n x n x T arrays) that the rest of the
package passes around, so callers never have to keep orderings aligned.
"""
from collections import namedtuple
import itertools

import networkx as nx
import numpy as np

from .utils import intersection, zigzag_persistence
from .params import interval_stats
from .topfunctions import adjacency_correlation, autocorrelation

NetworkStats = namedtuple('NetworkStats', ['density', 'rate'])


def normalize_intervals(ints, T):
  #sort, clip to [0,T], drop empty intervals and merge overlapping or touching ones
  out = []
  for (s, e) in sorted((float(s), float(e)) for (s, e) in ints):
    s = max(s, 0.0)
    e = min(e, float(T))
    if e <= s:
      continue
    if len(out) > 0 and s <= out[-1][1]:
      out[-1] = (out[-1][0], max(out[-1][1], e))
    else:
      out.append((s, e))
  return out


class TemporalNetwork:
  """Nodes present on [0, T]; each edge present on a list of disjoint intervals.

  Build one with TemporalNetwork.from_intervals, from_array or from_contacts,
  or with a model from tempnets.generators.
  """

  def __init__(self, nodes, intervals, T):
    self.T = T
    self.nodes = sorted(set(nodes) | {v for e in intervals for v in e})
    self.intervals = {}
    for (u, v), ints in intervals.items():
      if u == v:
        raise ValueError("self-loop on node %r" % (u,))
      key = (min(u, v), max(u, v))
      ints = normalize_intervals(list(ints) + self.intervals.get(key, []), T)
      if len(ints) > 0:
        self.intervals[key] = ints

  @classmethod
  def from_intervals(cls, intervals, T, nodes=()):
    """intervals: {(u, v): [(start, end), ...]}. Nodes default to the edge endpoints."""
    return cls(nodes, intervals, T)

  @classmethod
  def from_array(cls, arr):
    """A discrete-time network: arr is n x n x T and arr[i, j, t] != 0 when i and j are
    joined at step t. Step t covers the time interval [t - 1/2, t + 1/2], so runs of
    consecutive steps become one interval."""
    n, T = arr.shape[0], arr.shape[2]
    on = (arr != 0) | (arr.transpose(1, 0, 2) != 0)
    intervals = {}
    for i in range(n):
      for j in range(i + 1, n):
        steps = np.flatnonzero(on[i, j])
        if len(steps) > 0:
          intervals[(i, j)] = [(t - 0.5, t + 0.5) for t in steps]
    return cls(range(n), intervals, T)

  @classmethod
  def from_contacts(cls, contacts, duration, T=None, nodes=()):
    """contacts: iterable of (u, v, t). Each contact keeps its edge present on
    [t, t + duration]; overlapping contacts merge. T defaults to the last contact's end."""
    intervals = {}
    end = 0
    for (u, v, t) in contacts:
      key = (min(u, v), max(u, v))
      intervals.setdefault(key, []).append((t, t + duration))
      end = max(end, t + duration)
    return cls(nodes, intervals, end if T is None else T)

  @property
  def n(self):
    return len(self.nodes)

  @property
  def edges(self):
    return list(self.intervals.keys())

  def __repr__(self):
    return "TemporalNetwork(n=%d, T=%g, edges=%d)" % (self.n, self.T, len(self.intervals))

  def graph(self):
    """The networkx graph of every edge present at some time."""
    G = nx.Graph()
    G.add_nodes_from(self.nodes)
    G.add_edges_from(self.intervals.keys())
    return G

  def level_set(self, t):
    """The networkx graph of the edges present at time t."""
    H = nx.Graph()
    H.add_nodes_from(self.nodes)
    H.add_edges_from(e for e, ints in self.intervals.items() if any(s <= t < x for (s, x) in ints))
    return H

  def stats(self):
    """Achieved (density, rate): the fraction of node pairs joined, averaged over [0, T],
    and new edges per node pair per unit time."""
    return NetworkStats(*interval_stats(self.intervals, self.n, self.T))

  def to_array(self, times=None):
    """n x n x len(times) array sampling the network at times (default 0, 1, ..., T-1);
    nodes are in the order of self.nodes."""
    times = np.arange(int(np.ceil(self.T))) if times is None else np.asarray(times)
    index = {v: i for i, v in enumerate(self.nodes)}
    arr = np.zeros(shape=(self.n, self.n, len(times)))
    for (u, v), ints in self.intervals.items():
      on = np.zeros(len(times), dtype=bool)
      for (s, e) in ints:
        on |= (s <= times) & (times < e)
      arr[index[u], index[v], :] = on
      arr[index[v], index[u], :] = on
    return arr

  def flag(self, max_dim=2):
    """The temporal clique (flag) complex: every clique of up to max_dim + 1 nodes, present
    on the intervals where all of its edges are present."""
    cx = {(v,): [(0.0, float(self.T))] for v in self.nodes}
    for clique in nx.enumerate_all_cliques(self.graph()):
      if len(clique) > max_dim + 1:
        break
      if len(clique) == 1:
        continue
      sx = tuple(sorted(clique))
      if len(sx) == 2:
        ints = self.intervals[sx]
      else:
        ints = intersection([self.intervals[b] for b in itertools.combinations(sx, 2)])
      if len(ints) > 0:
        cx[sx] = ints
    return TemporalComplex(self.nodes, cx, self.T)

  def graph_autocorrelation(self, snapshots=100, lag=1):
    """Autocorrelation of the adjacency correlation between consecutive level sets.
    snapshots: the times to sample, or a number of evenly spaced times in [0, T]."""
    if np.isscalar(snapshots):
      snapshots = np.linspace(0, self.T, int(snapshots))
    gamma = []
    prev = self.level_set(snapshots[0])
    for t in snapshots[1:]:
      cur = self.level_set(t)
      gamma.append(adjacency_correlation(prev, cur))
      prev = cur
    return autocorrelation(gamma, lag)


class TemporalComplex:
  """A zigzag simplicial complex on [0, T]: each simplex (a sorted tuple of nodes)
  present on a list of disjoint intervals, always inside its faces' intervals."""

  def __init__(self, nodes, intervals, T):
    self.nodes = list(nodes)
    self.T = T
    #faces before cofaces, as dionysus requires
    self.intervals = dict(sorted(intervals.items(), key=lambda kv: len(kv[0])))

  @property
  def simplices(self):
    return list(self.intervals.keys())

  @property
  def dim(self):
    return max(len(sx) for sx in self.intervals) - 1

  def __repr__(self):
    counts = [sum(1 for sx in self.intervals if len(sx) == k + 1) for k in range(self.dim + 1)]
    return "TemporalComplex(T=%g, simplices by dimension=%s)" % (self.T, counts)

  def to_dionysus(self):
    """(simplices as lists of vertex indices, entry/exit lists) for d.zigzag_homology_persistence."""
    index = {v: i for i, v in enumerate(self.nodes)}
    sxs = [[index[v] for v in sx] for sx in self.intervals]
    dtimes = [list(itertools.chain.from_iterable(ints)) for ints in self.intervals.values()]
    return sxs, dtimes

  def persistence(self):
    """Zigzag persistence diagrams: one array of (birth, death) rows per dimension."""
    sxs, dtimes = self.to_dionysus()
    zz, dgms, cells = zigzag_persistence(sxs, dtimes)
    out = [np.array([(p.birth, p.death) for p in dgm], dtype=float).reshape(-1, 2) for dgm in dgms]
    while len(out) < self.dim + 1:
      out.append(np.zeros((0, 2)))
    return out
