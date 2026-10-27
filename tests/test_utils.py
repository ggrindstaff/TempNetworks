import networkx as nx
import numpy as np
import dionysus as d
import pytest

from tempnets.utils import (level_set, intersection, time_list_to_ee, graph_ee, to_ee,
                            ER_periodic, periodic_intervals, exponential_rewiring,
                            topological_rewiring, remove_a_loop, add_rand_edge, remove_rand_edge,
                            plot_diags, temp_net_to_clique_clx)
from tempnets.topfunctions import autocorrelation


def path_graph():
  G = nx.Graph()
  G.add_edges_from([(0, 1), (1, 2)])
  return G

def strictly_increasing(ee):
  return all(a < b for a, b in zip(ee, ee[1:]))


def test_level_set_closed_intervals():
  G = path_graph()
  times = [[0, 100]] * 3 + [[20, 30], [0, 10]]
  assert sorted(level_set(G, times, 5).edges) == [(1, 2)]
  assert sorted(level_set(G, times, 25).edges) == [(0, 1)]

def test_level_set_open_ended_interval_does_not_leak():
  G = path_graph()
  #edge (0,1) is present from 20 onwards; edge (1,2) on (0,10)
  times = [[0, 100]] * 3 + [[0, 10, 20], [0, 10]]
  assert sorted(level_set(G, times, 25).edges) == [(0, 1)]
  assert sorted(level_set(G, times, 15).edges) == []
  assert sorted(level_set(G, times, 5).edges) == [(0, 1), (1, 2)]

def test_graph_ee_matches_graph_edge_order():
  np.random.seed(0)
  net, sxs, times = ER_periodic(6, 0.9, 5, 2, 30, 2)
  ee = graph_ee(net, times, 30)
  for t in np.linspace(0, 30, 61):
    H = level_set(net, ee, t)
    for (u, v) in net.edges:
      on = any(a < t < b for (a, b) in times[(min(u, v), max(u, v))])
      if on:
        assert H.has_edge(u, v)


def test_intersection():
  assert intersection([[(0, 10)], [(1, 2), (5, 6)]]) == [(1, 2), (5, 6)]
  assert intersection([[(0, 10)], [(2, 3)], [(2.5, 9)]]) == [(2.5, 3)]
  #touching intervals do not intersect
  assert intersection([[(0, 1)], [(1, 2)]]) == []

def test_time_list_to_ee():
  assert time_list_to_ee([1, 2, 3, 7], 10) == [0.5, 3.5, 6.5, 7.5]
  with pytest.raises(ValueError):
    time_list_to_ee([1, 2], 10, inter='bogus')


def test_periodic_intervals_never_overlap():
  np.random.seed(3)
  for _ in range(50):
    ints = periodic_intervals(100, 0.5, 3)
    assert strictly_increasing([x for iv in ints for x in iv])
  with pytest.raises(ValueError):
    periodic_intervals(100, -5, 15)

def test_exponential_rewiring_gives_valid_zigzag():
  np.random.seed(1)
  G = nx.watts_strogatz_graph(12, 4, 0)
  net, sxs, times = exponential_rewiring(G, 0.01, 5, 2)
  dtimes = to_ee(times, sxs, 5)
  assert all(strictly_increasing(ee) for ee in dtimes)
  d.zigzag_homology_persistence(d.Filtration(sxs), dtimes)


def test_remove_a_loop_removes_a_cycle_edge():
  np.random.seed(0)
  G = nx.cycle_graph(5)
  e = remove_a_loop(G)
  assert e is not None and not G.has_edge(*e)
  assert remove_a_loop(nx.path_graph(4)) is None

def test_add_and_remove_rand_edge():
  np.random.seed(0)
  G = nx.path_graph(4)
  e = remove_rand_edge(G)
  assert not G.has_edge(*e)
  for _ in range(20):
    H = nx.empty_graph(3)
    H.add_edge(0, 1)
    assert add_rand_edge(H, exclude=(0, 2)) == (1, 2)

def test_topological_rewiring_gives_valid_zigzag():
  np.random.seed(2)
  G = nx.gnp_random_graph(12, 0.4, seed=2)
  net, sxs, times = topological_rewiring(G, 0.01, 5)
  dtimes = to_ee(times, sxs, 5)
  assert all(strictly_increasing(ee) for ee in dtimes)
  d.zigzag_homology_persistence(d.Filtration(sxs), dtimes)


class Point:
  def __init__(self, birth, death):
    self.birth = birth
    self.death = death

def test_plot_diags_with_empty_dimension(tmp_path):
  dgms = [[Point(0, 5), Point(1, float('inf'))], [], [Point(2, 3)]]
  plot_diags(dgms, 20, save=str(tmp_path / "dgm"))
  assert (tmp_path / "dgm.png").exists()


def test_autocorrelation():
  g = [1, 2, 3, 4, 5, 6]
  gbar = np.mean(g)
  denom = sum((x - gbar) ** 2 for x in g)
  for lag in (1, 2, 3):
    num = sum((g[t] - gbar) * (g[t + lag] - gbar) for t in range(len(g) - lag))
    assert autocorrelation(g, lag) == pytest.approx(num / denom)


def contacts_array(contacts, n, T):
  arr = np.zeros((n, n, T))
  for (i, j, t) in contacts:
    arr[i, j, t] = 1
  return arr

def test_temp_net_to_clique_clx():
  #triangle 0-1-2 with (0,2) only at t=4 and (1,2) only at t=5: the edges touch at
  #t=4.5 but are never on together, so there must be no triangle
  contacts = [(0, 1, t) for t in range(3, 7)] + [(0, 2, 4), (1, 2, 5)]
  sxs, times = temp_net_to_clique_clx(contacts_array(contacts, 3, 10))
  assert [0, 1, 2] not in sxs
  #now (1,2) overlaps (0,2): triangle on exactly the common interval
  #(stored as (2,1) to check both directions are read)
  contacts = [(0, 1, t) for t in range(3, 7)] + [(0, 2, 4), (0, 2, 5), (2, 1, 5)]
  sxs, times = temp_net_to_clique_clx(contacts_array(contacts, 3, 10))
  assert times[sxs.index([0, 1, 2])] == [4.5, 5.5]
  assert all(strictly_increasing(ee) for ee in times)
  d.zigzag_homology_persistence(d.Filtration(sxs), times)
