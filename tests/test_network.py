import numpy as np
import pytest

import tempnets as tn
from tempnets.network import normalize_intervals
from tempnets.discrete import rand_binomial
from tempnets.utils import array_homology, graph_ee, zigzag_persistence
from tempnets.topfunctions import graph_autocorrelation, pers_burstiness, top_burstiness, top_autocorr


def points(dgm):
  return sorted(map(tuple, tn.utils.as_arrays([dgm])[0].tolist()))


def test_normalize_intervals():
  assert normalize_intervals([(5, 6), (-1, 2), (1, 3), (6, 7), (8, 8), (9, 12)], 10) == \
      [(0, 3), (5, 7), (9, 10)]

def test_from_intervals_canonical_edges():
  net = tn.TemporalNetwork.from_intervals({(2, 1): [(0, 1)], (1, 2): [(1, 3)], (0, 3): [(4, 4)]}, 10)
  assert net.intervals == {(1, 2): [(0, 3)]}
  assert net.nodes == [0, 1, 2, 3]
  with pytest.raises(ValueError):
    tn.TemporalNetwork.from_intervals({(1, 1): [(0, 1)]}, 10)

def test_from_array_round_trip():
  np.random.seed(0)
  arr = rand_binomial(8, 30, 0.2, 0.6, init=0.3)
  net = tn.TemporalNetwork.from_array(arr)
  assert (net.to_array() == arr).all()

def test_from_contacts():
  net = tn.TemporalNetwork.from_contacts([(1, 0, 0), (0, 1, 1), (0, 1, 10), (2, 1, 3)], duration=2)
  assert net.intervals == {(0, 1): [(0, 3), (10, 12)], (1, 2): [(3, 5)]}
  assert net.T == 12


def test_flag_matches_array_homology():
  np.random.seed(1)
  arr = rand_binomial(10, 40, 0.15, 0.7, init=0.3)
  dgms = tn.TemporalNetwork.from_array(arr).flag(max_dim=2).persistence()
  old = array_homology(arr)
  assert len(dgms) == len(old)
  for new_dgm, old_dgm in zip(dgms, old):
    assert points(new_dgm) == points(old_dgm)

def test_flag_intervals():
  net = tn.TemporalNetwork.from_intervals({(0, 1): [(0, 10)], (1, 2): [(2, 8)], (0, 2): [(5, 9)],
                                           (2, 3): [(0, 1)]}, 10)
  cx = net.flag(max_dim=2)
  assert cx.intervals[(0, 1, 2)] == [(5, 8)]
  assert cx.dim == 2
  #faces come before cofaces
  assert [len(sx) for sx in cx.simplices] == sorted(len(sx) for sx in cx.simplices)

def test_string_node_labels():
  net = tn.TemporalNetwork.from_intervals({('a', 'b'): [(0, 5)], ('b', 'c'): [(1, 4)], ('a', 'c'): [(2, 6)]}, 10)
  dgms = net.flag().persistence()
  #one component for the whole interval and the triangle never creates a 1-cycle
  assert len(dgms[1]) == 0


def test_graph_autocorrelation_matches_function():
  net = tn.generators.er_periodic(15, 50, 0.3, 0.1, seed=2)
  snapshots = np.linspace(0, 50, 60)
  G = net.graph()
  assert net.graph_autocorrelation(snapshots) == \
      pytest.approx(graph_autocorrelation(G, graph_ee(G, net.intervals, 50), snapshots))

def test_level_set():
  net = tn.TemporalNetwork.from_intervals({(0, 1): [(0, 2), (5, 6)], (1, 2): [(1, 5)]}, 10)
  assert sorted(net.level_set(1.5).edges) == [(0, 1), (1, 2)]
  assert sorted(net.level_set(5).edges) == [(0, 1)]
  assert sorted(net.level_set(7).edges) == []


def test_stats_accept_both_diagram_formats():
  net = tn.generators.er_periodic(12, 40, 0.3, 0.1, seed=3)
  cx = net.flag()
  sxs, dtimes = cx.to_dionysus()
  zz, raw, cells = zigzag_persistence(sxs, dtimes)
  arrays = cx.persistence()
  s = np.linspace(0, 40, 50)
  assert top_autocorr(arrays, s) == pytest.approx(top_autocorr(raw, s, format='dionysus'), rel=1e-4)
  assert pers_burstiness(arrays, 40) == pytest.approx(pers_burstiness(raw, 40), rel=1e-4, nan_ok=True)
  #arrays carry exact times, dionysus float32 ones
  assert top_burstiness(arrays, 40) == pytest.approx(top_burstiness(raw, 40), rel=1e-4, nan_ok=True)


@pytest.mark.parametrize("model", ["binomial", "er_periodic", "ws_rewiring", "topological_rewiring"])
def test_generators_match_targets_and_seed(model):
  f = getattr(tn.generators, model)
  state = np.random.get_state()
  net = f(30, 200, 0.3, 0.1, seed=5)
  #seeding does not disturb the global random state
  assert (np.random.get_state()[1] == state[1]).all()
  assert f(30, 200, 0.3, 0.1, seed=5).intervals == net.intervals
  assert f(30, 200, 0.3, 0.1, seed=6).intervals != net.intervals
  density, rate = net.stats()
  assert abs(density - 0.3) < 0.03
  assert rate == pytest.approx(0.1, rel=0.1)

def test_poisson_generator():
  density, rate = tn.generators.poisson(30, 200, 0.3, seed=0).stats()
  assert density == pytest.approx(0.3, rel=0.05)

def test_persistence_reports_exact_times():
  #96.57 hours is not a float32; deaths at T must still equal T
  T = 347660 / 3600
  net = tn.TemporalNetwork.from_intervals({(0, 1): [(1 / 3, 50.1)], (1, 2): [(2.7, T)]}, T)
  dgms = net.flag().persistence()
  assert T in dgms[0][:, 1].tolist()
  #every reported time is exactly one of the event times
  assert set(dgms[0].ravel().tolist()) <= {0.0, 1 / 3, 2.7, 50.1, T}
  assert 50.1 in dgms[0][:, 0].tolist()
