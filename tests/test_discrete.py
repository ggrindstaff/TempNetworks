import networkx as nx
import numpy as np

from tempnets.discrete import to_temp_net, partition, scrambly_arr, rref_hom, rand_binomial


def test_to_temp_net():
  net = nx.path_graph(3)
  times = {(0, 1): [(0, 2)], (1, 2): [(1.5, 3)], (0, 1, 2): [(1.5, 2)]}
  arr = to_temp_net(net, times, [0, 1, 2, 3])
  assert arr[:, :, 0].tolist() == [[0, 1, 0], [1, 0, 0], [0, 0, 0]]
  assert arr[1, 2].tolist() == [0, 0, 1, 1]
  assert (arr == arr.transpose(1, 0, 2)).all()

def test_partition():
  arr = np.zeros((2, 2, 6))
  arr[0, 1, [1, 4]] = 1
  new = partition(arr, [0, 2, 4])
  assert new[0, 1].tolist() == [1, 0, 1]
  #time labels instead of step indices; the last window is open-ended
  new = partition(arr, [0, 25], times=[0, 10, 20, 30, 40, 50])
  assert new[0, 1].tolist() == [1, 1]

def test_scrambly_arr_permutes_time():
  np.random.seed(0)
  arr = np.random.rand(3, 3, 8)
  new = scrambly_arr(arr)
  assert sorted(new[0, 1]) == sorted(arr[0, 1])
  for t in range(8):
    assert any((new[:, :, t] == arr[:, :, s]).all() for s in range(8))

def test_rref_hom():
  np.random.seed(0)
  arr = rand_binomial(6, 20, 0.2, 0.7, init=0.3)
  rref, results = rref_hom(arr, 3)
  assert len(rref) == 3 and len(results) == 3
