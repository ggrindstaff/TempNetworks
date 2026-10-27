"""Legacy: TemporalGraphComplex, an earlier object-oriented interface.

It needs bats-tda (pip install "tempnets[extras]"), which does not build on
current Python versions. tempnets.TemporalNetwork and tempnets.generators cover
the same ground.
"""
import networkx as nx
import bats
import numpy as np
import dionysus as d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rc
import itertools
from .utils import intersection, plot_diags, zigzag_persistence
rc('animation', html='jshtml')

class TemporalGraphComplex:
    tda = 'bats'

    def __init__(self,n=10,T=40,max_dim=2,**kwargs):
        self.n = n
        self.T = T
        self.max_dim = max_dim

        if 'tda' in kwargs:
            self.tda = kwargs['tda']
        else: self.tda = 'dionysus'

        if 'temporal_pattern' in kwargs:
          self.temporal_pattern = kwargs['temporal_pattern']
        else:
          self.set_poisson()

        if 'graph_type' in kwargs:
          self.graph_type =kwargs['graph_type']
        else:
          self.set_ER()

    def set_WS(self,k = 3,p=0):
      self.graph_type='WS'
      self.k = k
      self.p = p

    def set_ER(self,p=0.5):
      self.graph_type='ER'
      self.p = p

    def set_periodic(self,freq=8,l=2):
      self.temporal_pattern = 'periodic'
      self.freq = freq
      self.l = l

    def set_poisson(self,l=2):
      self.temporal_pattern = 'poisson'
      self.l = l

    def to_dionysus_format(self):
        all_K = self.X.complex().get_simplices()
        self.Y = d.Filtration(all_K)
        dtimes = []
        for sx in all_K:
            if len(sx)==1:
                dtimes.append([0,self.T])
            else:
                dtimes.append(list(itertools.chain.from_iterable(self.times[tuple(sx)])))
        return dtimes

    def periodic_intervals(self,tda = 'bats'):
        start = np.random.rand()*self.freq
        dur = np.random.exponential(self.l)
        t = [(start,start+dur)]
        while True:
            newstart = t[-1][1] + self.freq + np.random.normal(0,1)
            if newstart >= self.T: break
        #    dur = np.random.exponential(l)
            dur = self.l
            if newstart + dur <self.T:
              t.append((newstart,newstart + dur))
            #      print((newstart,newstart+dur))
            else:
              t.append((newstart,self.T))
              break
        return t

    def poisson_intervals(self):
      start = np.random.exponential(self.l)
      dur = np.random.exponential(self.l)
      t = [(start,start+dur)]
      while True:
            newstart = t[-1][1] + np.random.exponential(self.l)
            if newstart >= self.T: break
        #    dur = np.random.exponential(l)
            dur = np.random.exponential(self.l)
            if newstart + dur <self.T:
              t.append((newstart,newstart + dur))
            #      print((newstart,newstart+dur))
            else:
              t.append((newstart,self.T))
              break
      return t

    def ER_edges(self,timestamps):
        for i in range(self.n):
            for j in range(i):
                roll = np.random.rand()
                if roll < self.p:
                    self.G.add_edge(j,i)
                    if timestamps=='periodic':
                      t = self.periodic_intervals()
                    if timestamps=='poisson':
                      t = self.poisson_intervals()
                    self.times[(j,i)] = t
                    for (t1,t2) in t:
                        self.X.add(t1,t2,[j,i])

    def ER_periodic(self):
        self.G = nx.Graph(directed=False)
        self.X = bats.ZigzagSimplicialComplex()
        self.times = {}
        #vertices remain in the network for the entire interval
        for i in range(self.n):
            self.X.add(0,self.T,[i])
            self.G.add_node(i)

        #add in edges via Erdos-Renyi uniform random
        self.ER_edges(timestamps='periodic')
        #turn network into clique complex
        self.flag()

    def init_clx(self):
        self.G = nx.Graph(directed=False)
        self.X = bats.ZigzagSimplicialComplex()
        self.times = {}
        #vertices remain in the network for the entire interval
        for i in range(self.n):
            self.X.add(0,self.T,[i])
            self.G.add_node(i)

        #add in edges via Erdos-Renyi uniform random
        if self.graph_type=='ER':
          self.ER_edges(timestamps=self.temporal_pattern)

        if self.graph_type=='WS':
          self.WS_exponential()
        #turn network into clique complex
        self.flag()

    def level_set_graph(self,t):
        H = nx.Graph()
        H.add_nodes_from(self.G)
        for edge in self.G.edges:
            ints = self.times[edge]
            for (b,d) in ints:
                if b<t and t<d:
                    H.add_edge(edge[0],edge[1])
        return H

    def intersection(self,all_times):
        return intersection(all_times)

    def add_n_simplices(self,n=2):
        m = self.X.ncells(0)
        e = self.X.complex().get_simplices(1)
        simplices = list(itertools.combinations(range(m), n+1))
        for simplex in simplices:
            boundary1 = list(itertools.combinations(simplex,2))
            if all([list(b) in e for b in boundary1]):
                time = TemporalGraphComplex.intersection(self,[self.times[b] for b in boundary1])
                if len(time)>0:
                  self.times[simplex] = time
                  for (u,v) in time:
                    self.X.add_recursive(u,v,list(simplex))

    def flag(self):
        for n in range(2,self.max_dim+1):
            self.add_n_simplices(n)

    def level_set(self,t):
        H = nx.Graph()
        H.add_nodes_from(self.G)
        for edge in self.G.edges:
            ints = self.times[edge]
            for (b,d) in ints:
                if b<t and t<d:
                    H.add_edge(edge[0],edge[1])
        return H

    def is_in(self,time,t):
        flag = False
        for (t1,t2) in time:
            if t1<=t<=t2:
                flag=True
        return flag

    def next_time(self):
      if self.temporal_pattern=='poisson':
        return np.random.exponential(self.l)

    def WS_exponential(self):
        self.G = nx.watts_strogatz_graph(self.n,self.k,self.p)
        self.X = bats.ZigzagSimplicialComplex()
        for i in range(self.n):
          self.X.add(0,self.T,[i])
        self.times = {}
        for edge in self.G.edges():
          self.times[edge] = [(0,self.T)]
        t = self.next_time()
        num = len(self.G.edges())
        G = self.G.copy()

        while t < self.T:
          lineup = list(G.edges())
          togo = lineup[np.random.randint(num)]
          G.remove_edge(togo[0],togo[1])
          recent = self.times[togo][-1]
          self.times[togo][-1] = (recent[0],t)
          stay = togo[np.random.randint(2)]
          new_neighbor = np.random.randint(self.n)
          while new_neighbor == stay or (stay,new_neighbor) in G.edges() or (new_neighbor,stay) in G.edges() or (min(stay,new_neighbor),max(stay,new_neighbor))==togo:
            new_neighbor = np.random.randint(self.n)
          G.add_edge(stay,new_neighbor)
          u = min(stay,new_neighbor)
          v = max(stay,new_neighbor)
          if (u,v) in self.times.keys():
            if self.times[(u,v)][-1][1]==self.T:
              print(str((u,v)) + "bad bad no good")
            self.times[(u,v)].append((t,self.T))
          else:
            self.times[(u,v)] =[(t,self.T)]
          t = t + self.next_time()

        for edge in self.times.keys():
          for (t0,t1) in self.times[edge]:
            self.X.add(t0,t1,[edge[0],edge[1]])


    def dionysus_ph(self):
      dtimes = self.to_dionysus_format()
      zz,dgms,cell=zigzag_persistence(self.Y,dtimes)
      self.dgms = dgms
      self.birth = []
      self.death = []
      self.all_finite_pers = []
      for i,dgm in enumerate(dgms):
        newbirth = []
        newdeath = []
        newpers = []
        for p in dgm:
          newbirth.append(p.birth)
          newdeath.append(p.death)
          if p.death < self.T:
            newpers.append(p.death-p.birth)
        self.birth.append(newbirth)
        self.death.append(newdeath)
        self.all_finite_pers.append(newpers)

    def bats_ph(self):
      self.ps = bats.ZigzagBarcode(self.X, 1, bats.F2())
      #needs more processing

    def compute_homology(self):
      self.init_clx()
      if self.tda=='dionysus':
        self.dionysus_ph()
      if self.tda=='bats':
        self.bats_ph()

    def plot_homology(self):
      plot_diags(self.dgms,self.T)
