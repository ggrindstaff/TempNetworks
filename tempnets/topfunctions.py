"""Statistics of temporal networks and their zigzag persistence diagrams:
autocorrelation (snapshot and snapshot-free), burstiness and related measures.
Diagram arguments accept any format (dionysus diagrams, (birth, death) tuples or
arrays).
"""
import numpy as np
import networkx as nx
from .utils import level_set, as_arrays
import dionysus as d

def autocorrelation(gamma,lag=1):
  """Sample autocorrelation of the series gamma at the given lag:
  sum_{t < n-lag} (gamma_t - mean)(gamma_{t+lag} - mean) / sum_t (gamma_t - mean)^2,
  or 0 for a constant series."""
  gamma_bar = sum(gamma)/len(gamma)
  num= 0
  denom=0
  for t in range(len(gamma)-lag):
    num = num + (gamma[t]-gamma_bar)*(gamma[t+lag]-gamma_bar)
  for t in range(len(gamma)):
    denom = denom + (gamma[t]-gamma_bar)**2
  if denom == 0:
    return 0
  else:
    return num/denom
  
def adjacency_correlation(G,H):
  """Adjacency correlation of two graphs on the same nodes: for each node, the
  number of neighbours shared by G and H over sqrt(degree in G * degree in H)
  (0 for a node isolated in either), averaged over nodes."""
  A = nx.to_numpy_array(G)
  B = nx.to_numpy_array(H)
  n = len(G.nodes)
  gamma = np.zeros(n)
  for i in range(n):
    if np.sqrt(sum(A[i,]))*np.sqrt(sum(B[i,]))==0:
      pass
    else:
      gamma[i] = sum(A[i,]*B[i,])/(np.sqrt(sum(A[i,]))*np.sqrt(sum(B[i,])))
  return sum(gamma)/n

def graph_autocorrelation(G,times, snapshots):
  """Lag-1 autocorrelation of the adjacency correlation between the level sets of
  G at consecutive snapshots; times is in level_set's format. See
  TemporalNetwork.graph_autocorrelation and graph_autocorrelation_continuous."""
  gamma = [adjacency_correlation(level_set(G,times,snapshots[i]),
                                 level_set(G,times,snapshots[i+1])) for i in range(len(snapshots)-1)]
  return autocorrelation(gamma)

def dgm_points(dgms,format=None):
  """Births and deaths of the points of all dimensions, as two arrays. Accepts
  any diagram format; format is accepted for compatibility and ignored."""
  pts = np.concatenate(as_arrays(dgms)+[np.zeros((0,2))])
  return pts[:,0],pts[:,1]

def dgm_correlation(dgms,t1,t2,format=None):
  """Correlation of a diagram between times t1 and t2: f12/sqrt(f1*f2), where f1
  and f2 count the points alive at t1 and t2 and f12 those alive at both (0 if f1 or
  f2 is 0). Points are alive on [birth, death]; any diagram format."""
  birth,death = dgm_points(dgms)
  a1 = (birth<=t1) & (t1<=death)
  a2 = (birth<=t2) & (t2<=death)
  f1 = a1.sum()
  f2 = a2.sum()
  if f1*f2==0:
    return 0
  return (a1 & a2).sum()/np.sqrt(f1*f2)

def top_autocorr(dgms,snapshots,lag=1,format=None):
  """Lag autocorrelation of dgm_correlation between consecutive snapshots (any
  diagram format). See top_autocorr_continuous for the snapshot-free version."""
  birth,death = dgm_points(dgms)
  s = np.asarray(snapshots,dtype=float)
  #alive[k,i]: point k is alive at snapshots[i]
  alive = (birth[:,None]<=s[None,:]) & (s[None,:]<=death[:,None])
  f = alive.sum(axis=0)
  f12 = (alive[:,:-1] & alive[:,1:]).sum(axis=0)
  f1f2 = f[:-1]*f[1:]
  gamma = np.where(f1f2==0, 0, f12/np.sqrt(np.maximum(f1f2,1)))
  return autocorrelation(list(gamma),lag)

"""## Snapshot-free autocorrelation

The snapshot statistics compare what is alive at consecutive snapshots t and t+tau
(tau being the snapshot spacing) and take the lag-1 autocorrelation of that series.
The versions below use every time t instead of a grid:
  gamma(t) = cosine similarity of what is alive at t and at t+tau,  t in [0, T-tau]
  rho = int_0^{T-2tau} (gamma(t)-gbar)(gamma(t+tau)-gbar) dt / int_0^{T-tau} (gamma(t)-gbar)^2 dt
with gbar the mean of gamma on [0, T-tau]. gamma is a step function, so the integrals
are computed exactly from its breakpoints; single instants carry no weight.
"""

def cover(starts,ends,x):
  """Number of intervals [starts[k], ends[k]] containing each x; starts and ends
  must be sorted."""
  return np.searchsorted(starts,x,side='right') - np.searchsorted(ends,x,side='left')

def cosine(f12,f1,f2):
  """f12/sqrt(f1*f2), or 0 where f1*f2 == 0."""
  f1f2 = f1*f2
  return np.where(f1f2==0, 0, f12/np.sqrt(np.maximum(f1f2,1)))

def continuous_autocorrelation(gamma,breaks,T,tau):
  """The snapshot-free autocorrelation rho (see above) of a step function gamma,
  vectorized in t and constant between the given breakpoints; needs 0 < 2*tau < T."""
  if not 0 < 2*tau < T:
    raise ValueError("need 0 < 2*tau < T")
  breaks = np.asarray(breaks,dtype=float)
  breaks = breaks[np.isfinite(breaks)]
  def segments(hi,extra):
    pts = np.unique(np.clip(np.concatenate([breaks,extra,[0,hi]]),0,hi))
    lens = np.diff(pts)
    return pts[:-1]+lens/2, lens
  mids,lens = segments(T-tau,[])
  g = gamma(mids)
  gbar = (g*lens).sum()/lens.sum()
  denom = ((g-gbar)**2*lens).sum()
  if denom==0:
    return 0
  mids,lens = segments(T-2*tau,breaks-tau)
  return ((gamma(mids)-gbar)*(gamma(mids+tau)-gbar)*lens).sum()/denom

def top_autocorr_continuous(dgms,tau,T):
  """Snapshot-free top_autocorr with lag tau on [0, T]; diagrams in any format."""
  birth,death = dgm_points(dgms)
  starts = np.sort(birth)
  ends = np.sort(death)
  #points alive at both t and t+tau are those with birth <= t <= death-tau
  both = death-tau >= birth
  bstarts = np.sort(birth[both])
  bends = np.sort(death[both]-tau)
  def gamma(t):
    return cosine(cover(bstarts,bends,t),cover(starts,ends,t),cover(starts,ends,t+tau))
  breaks = np.concatenate([birth,death,birth-tau,death-tau])
  return continuous_autocorrelation(gamma,breaks,T,tau)

def top_stationarity(dgms,times):
  """Heuristic test for a trend in a diagram: draws 10000 times t uniformly in
  [0, T] and windows w ~ Exp(1) (clipped at T), and returns the covariance between t
  and dgm_correlation(t, t + w). T is read from times, either an interval dict or a
  list of entry/exit lists."""
  if type(times)==dict:
    format='tuples'
    T = max([max(ent)[1] for ent in list(times.values())])

  if type(times)==list:
    format='dionysus'
    T = int(np.ceil(np.max(times)))

  runs = 10000
  t = []
  dac = []
  for x in range(runs):
    t.append(np.random.rand()*T)
    w = np.random.exponential(scale=1)
    if t[x]+w > T:
      w = T-t[x]
    corr = dgm_correlation(dgms,t[x],t[x]+w)
    dac.append(corr)

  return np.cov(t,dac)[0,1]

#  w1=

def cv(t):
  """Burstiness coefficient B = (sigma - mu)/(sigma + mu) of the values t
  (Goh and Barabasi): -1 for periodic, 0 for Poisson-like and towards 1 for bursty
  sequences. (Named cv for historical reasons.)"""
  s = np.std(t)
  m= np.average(t)
  return (s/m-1)/(s/m+1)

def burstiness(times):
  """Burstiness coefficient (see cv) of the inter-contact times of all edges: the
  gaps between consecutive presence intervals of each edge in the interval dict
  times (higher simplices are ignored)."""
  inter_contact=[]
  for e in times.keys():
    if len(e)==2:
      time = times[e]
      for i in range(len(time)-1):
        (u,v) = time[i]
        (w,x) = time[i+1]
        inter_contact.append(w-v)
  B = cv(inter_contact)
  return B

def top_change_burstiness(dgms):
  """Legacy helper, not used elsewhere: the burstiness coefficient (see cv) of
  the values in pers[0] and pers[1] over the entries pers of dgms, each a pair of
  lists."""
  ls = []
  for pers in dgms:
    ls = ls + pers[0]+pers[1]
  return cv(ls)

def pers_burstiness(dgms,T):
  """Burstiness coefficient (see cv) of the finite persistences (death - birth
  for deaths before T), pooled over dimensions; any diagram format."""
  all_finite_pers = []
  for i,dgm in enumerate(as_arrays(dgms)):
    newpers = []
    for (b,dth) in dgm:
      if dth < T:
        newpers.append(dth-b)
    all_finite_pers.append(newpers)
  return cv([pers for sublist in all_finite_pers for pers in sublist])

def top_burstiness(dgms,T, j=None):
  """Burstiness coefficients (see cv) of the gaps between consecutive birth
  times and between consecutive death times (deaths before T), for each of the first
  j dimensions (all by default): [births dim 0, deaths dim 0, births dim 1, ...]."""
  scores = []
  if j==None:
    j = len(dgms)
  for i,dgm in enumerate(as_arrays(dgms)):
    if i>=j:
      break
    newbirth = []
    newdeath = []
    for (b,dth) in dgm:
      newbirth.append(b)
      if dth < T:
        newdeath.append(dth)
    interbirth = np.diff(sorted(newbirth))
    interdeath = np.diff(sorted(newdeath))
    scores.append(cv(interbirth))
    scores.append(cv(interdeath))
  return scores

def persistence(dgms,t1,t2):
  """(persist, change) between times t1 and t2: the number of points alive
  throughout [t1, t2], and the number of births plus deaths within it."""
  persist=0
  change=0
  for dgm in as_arrays(dgms):
    for (b,dth) in dgm:
      if b<=t1 and dth>=t2:
        persist=persist+1
      if t1<=b<=t2:
        change = change+1
      if t1<=dth<=t2:
        change = change+1
  return persist,change

def wasserstein_corr(dgms,max_dim=1):
  """Pairwise Wasserstein distances between the diagrams of m networks: dgms
  is a list of m diagram lists (dionysus format); returns an array
  (max_dim, m, m) with one distance matrix per dimension below max_dim."""
  m = len(dgms)
  inds = range(m)
  corrdist = np.ndarray(shape=(max_dim,m,m))
  for j in range(max_dim):
    corrdist[j] = [[d.wasserstein_distance(dgms[key][j],dgms[key1][j]) for key1 in inds] for key in inds]
  return corrdist
