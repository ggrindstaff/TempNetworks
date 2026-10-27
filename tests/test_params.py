from math import exp

import networkx as nx
import numpy as np
import pytest

from tempnets.params import (binomial_params, poisson_params, er_periodic_params, ws_params,
                             rewiring_interval, interval_stats, array_stats)
from tempnets.utils import ER_periodic, exponential_rewiring, topological_rewiring
from tempnets.discrete import rand_binomial, rand_poisson

DENS = 0.3
RATE = 0.1


def test_rand_binomial_matches_targets():
  np.random.seed(0)
  birth, stay = binomial_params(DENS, RATE)
  arr = rand_binomial(30, 400, birth, stay, init=DENS)
  assert (arr == arr.transpose(1, 0, 2)).all()
  dens, rate = array_stats(arr)
  assert dens == pytest.approx(DENS, rel=0.05)
  assert rate == pytest.approx(RATE, rel=0.05)

def test_binomial_params_infeasible():
  with pytest.raises(ValueError):
    binomial_params(0.1, 0.2)

def test_rand_poisson_density():
  np.random.seed(0)
  lam = poisson_params(DENS)
  assert (1 - exp(-lam)) / lam == pytest.approx(DENS)
  arr = rand_poisson(30, 200, lam)
  assert arr.shape == (30, 30, 200)
  assert (arr == arr.transpose(1, 0, 2)).all()
  dens, rate = array_stats(arr)
  assert dens == pytest.approx(DENS, rel=0.05)


def test_er_periodic_params_match_targets():
  np.random.seed(0)
  n, T, p = 30, 400, 2 * DENS
  freq, l = er_periodic_params(DENS, RATE, p)
  net, sxs, times = ER_periodic(n, p, freq, l, T, 1)
  dens, rate = interval_stats(times, n, T)
  assert dens == pytest.approx(DENS, rel=0.1)
  assert rate == pytest.approx(RATE, rel=0.1)

def test_er_periodic_params_infeasible():
  with pytest.raises(ValueError):
    er_periodic_params(DENS, RATE, DENS / 2)


def test_ws_params_match_targets():
  np.random.seed(0)
  n, T = 20, 50
  k, l = ws_params(n, DENS, RATE)
  assert k % 2 == 0
  G = nx.watts_strogatz_graph(n, k, 0)
  net, sxs, times = exponential_rewiring(G, l, T, 1)
  dens, rate = interval_stats(times, n, T)
  assert dens == pytest.approx(k / (n - 1))
  assert abs(dens - DENS) <= 1 / (n - 1)
  assert rate == pytest.approx(RATE, rel=0.1)

def test_topological_rewiring_rate():
  np.random.seed(0)
  n, T = 20, 50
  G = nx.gnp_random_graph(n, DENS, seed=0)
  net, sxs, times = topological_rewiring(G, rewiring_interval(n, RATE), T, 1)
  dens, rate = interval_stats(times, n, T)
  assert dens == pytest.approx(G.number_of_edges() / (n * (n - 1) / 2))
  assert rate == pytest.approx(RATE, rel=0.1)
