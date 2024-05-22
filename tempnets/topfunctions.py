import numpy as np
import networkx as nx
from .utils import level_set
import dionysus as d

def autocorrelation(gamma,lag=1):
  gamma_bar = sum(gamma)/len(gamma)
  num= 0
  denom=0
  for t in range(len(gamma)-lag):
    num = num + (gamma[t]-gamma_bar)*(gamma[t+lag]-gamma_bar)
    denom = denom + (gamma[t]-gamma_bar)**2
  denom = denom + (gamma[-lag]-gamma_bar)**2
  if denom == 0:
    return 0
  else:
    return num/denom
  
def adjacency_correlation(G,H):
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
  gamma = [adjacency_correlation(level_set(G,times,snapshots[i]),
                                 level_set(G,times,snapshots[i+1])) for i in range(len(snapshots)-1)]
  return autocorrelation(gamma)

def dgm_correlation(dgms,t1,t2,format='tuples'):
  f1 = 0
  f2 = 0
  f12 = 0
  if format=='tuples':
    for dgm in dgms:
      for (b,d) in dgm:
        f1 = f1 + int(b<=t1<=d)
        f2 = f2 + int(b<=t2<=d)
        f12 = f12 + int(b<=t1<=d and b<=t2<=d)
  if format=='dionysus':
    for dgm in dgms:
      for p in dgm:
        bi = p.birth
        de = p.death
        f1 = f1 + int(bi<=t1<=de)
        f2 = f2 + int(bi<=t2<=de)
        f12 = f12 + int(bi<=t1<=de and bi<=t2<=de)
  if f1*f2==0:
    return 0
  else: return f12/np.sqrt(f1*f2)

def top_autocorr(dgms,snapshots,lag=1,format='tuples'):
  gamma = [dgm_correlation(dgms,snapshots[i],snapshots[i+1],format) for i in range(len(snapshots)-1)]
  return autocorrelation(gamma,lag)

def top_stationarity(dgms,times):
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
    if t[x]+w > 40:
      w = 40-t[x]
    corr = dgm_correlation(dgms,t[x],t[x]+w,format='dionysus')
    dac.append(corr)

  return np.cov(t,dac)[0,1]

#  w1=

def cv(t):
  s = np.std(t)
  m= np.average(t)
  return (s/m-1)/(s/m+1)

def burstiness(times):
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
  ls = []
  for pers in dgms:
    ls = ls + pers[0]+pers[1]
  return cv(ls)

def pers_burstiness(dgms,T):
  all_finite_pers = []
  for i,dgm in enumerate(dgms):
    newpers = []
    for p in dgm:
      if p.death < T:
        newpers.append(p.death-p.birth)
    all_finite_pers.append(newpers)
  return cv([pers for sublist in all_finite_pers for pers in sublist])

def top_burstiness(dgms,T, j=None):
  scores = []
  if j==None:
    j = len(dgms)
  for i,dgm in enumerate(dgms):
    if i>=j:
      break
    newbirth = []
    newdeath = []
    for p in dgm:
      newbirth.append(p.birth)
      if p.death < T:
        newdeath.append(p.death)
    interbirth = np.diff(sorted(newbirth))
    interdeath = np.diff(sorted(newdeath))
    scores.append(cv(interbirth))
    scores.append(cv(interdeath))
  return scores

def persistence(dgms,t1,t2):
  persist=0
  change=0
  for dgm in dgms:
    for point in dgm:
      if point.birth<=t1 and point.death>=t2:
        persist=persist+1
      if t1<=point.birth<=t2:
        change = change+1
      if t1<=point.death<=t2:
        change = change+1
  return persist,change

def wasserstein_corr(dgms,max_dim=1):
  m = len(dgms)
  inds = range(m)
  corrdist = np.ndarray(shape=(max_dim,m,m))
  for j in range(max_dim):
    corrdist[j] = [[d.wasserstein_distance(dgms[key][j],dgms[key1][j]) for key1 in inds] for key in inds]
  return corrdist
