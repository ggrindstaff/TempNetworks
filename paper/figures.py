"""Figures for the paper.

Run from the repository root:  python paper/figures.py [fig4] [fig6] [fig11]
(all three by default). Each figure is written to paper/ as PDF and PNG. Slow
computations are cached in paper/cache/; delete a cache file to recompute it.
"""
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import networkx as nx
import numpy as np
import pandas as pd

import tempnets as tn
from tempnets.utils import as_arrays
from tempnets.topfunctions import top_burstiness

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
DATA = HERE.parent / "data" / "detailed_list_of_contacts_Hospital.dat"

TEXTWIDTH = 6.3  # inches
DIM_COLORS = ["#1f77b4", "#ff7f0e", "#7b3294"]
MODELS = [  # (generator name, label)
  ("binomial", "Binomial"),
  ("poisson", "Poisson"),
  ("er_periodic", "ER periodic"),
  ("ws_rewiring", "Watts-Strogatz rewiring"),
  ("topological_rewiring", "Topological self-inhibiting"),
  ("sun", "Periodic sun"),
]
MODEL_COLORS = dict(zip([m for m, _ in MODELS], ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974", "#222222"]))
DENSITY, RATE = 0.3, 0.1

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 200})


def save(fig, name):
  for ext in ("pdf", "png"):
    fig.savefig(HERE / ("%s.%s" % (name, ext)))
  plt.close(fig)
  print("wrote paper/%s.pdf and .png" % name)

def cached(name, compute):
  path = CACHE / (name + ".json")
  if path.exists():
    return json.loads(path.read_text())
  CACHE.mkdir(exist_ok=True)
  result = compute()
  path.write_text(json.dumps(result))
  return result

def make_model(name, n, T, seed):
  if name == "sun":
    return tn.generators.sun(n, T)
  if name == "poisson":
    return tn.generators.poisson(n, T, DENSITY, seed=seed)
  return getattr(tn.generators, name)(n, T, DENSITY, RATE, seed=seed)

def finite_persistence_max(dgms, T):
  #largest death - birth among classes that die before T
  pers = [d[:, 1] - d[:, 0] for d in as_arrays(dgms) if len(d) > 0]
  pers = np.concatenate(pers + [np.zeros(0)])
  deaths = np.concatenate([d[:, 1] for d in as_arrays(dgms) if len(d) > 0] + [np.zeros(0)])
  finite = pers[deaths < T]
  return finite.max() if len(finite) > 0 else 1.0

def draw_diagram(ax, dgms, T, legend=True, pmax=None):
  #rotated persistence diagram: birth against persistence (death - birth). Classes
  #still alive at T have unknown persistence and are drawn on the infinity line.
  #pmax sets the persistence axis, so several diagrams can share a scale.
  if pmax is None:
    pmax = finite_persistence_max(dgms, T)
  inf = 1.15 * pmax
  ax.axhline(inf, color="0.6", lw=0.7, ls="--", zorder=0)
  for dim, d in enumerate(as_arrays(dgms)):
    if len(d) > 0:
      pers = np.where(d[:, 1] >= T, inf, d[:, 1] - d[:, 0])
      ax.scatter(d[:, 0], pers, s=5, color=DIM_COLORS[dim % 3], alpha=0.6, lw=0, label="$H_%d$" % dim)
  ax.set_xlim(-0.03 * T, 1.03 * T)
  ax.set_ylim(-0.04 * pmax, 1.25 * pmax)
  ax.set_xticks([x for x in MaxNLocator(nbins=4, integer=True).tick_values(0, T) if 0 <= x <= T])
  yticks = [y for y in ax.yaxis.get_major_locator().tick_values(0, pmax) if 0 <= y <= pmax]
  ax.set_yticks(yticks + [inf])
  ax.set_yticklabels(["%g" % y for y in yticks] + [r"$\infty$"])
  ax.set_xlabel("birth")
  ax.set_ylabel("persistence")
  if legend:
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.92), frameon=False, handletextpad=0.1, markerscale=2)

def draw_snapshot(ax, net, t, pos):
  #the clique complex at time t: edges, with triangles shaded
  G = net.level_set(t)
  for tri in (c for c in nx.enumerate_all_cliques(G) if len(c) == 3):
    ax.fill(*zip(*[pos[v] for v in tri]), color="0.75", alpha=0.25, lw=0)
  nx.draw_networkx_edges(G, pos, ax=ax, width=0.5, edge_color="0.25")
  nx.draw_networkx_nodes(G, pos, ax=ax, node_size=6, node_color="0.1")
  ax.set_title("t = %g" % t, pad=1)
  ax.set_aspect("equal")
  ax.axis("off")


def fig4():
  """The six models at the same density and rate: snapshots of the clique complex and
  the persistence diagram."""
  n, T, seed = 15, 40, 3
  fig = plt.figure(figsize=(TEXTWIDTH, 8.2))
  subfigs = fig.subfigures(3, 2, hspace=0.04, wspace=0.04)
  nets = [make_model(name, n, T, seed) for name, _ in MODELS]
  all_dgms = [net.flag(max_dim=2).persistence() for net in nets]
  #one persistence scale for all six diagrams
  pmax = max(finite_persistence_max(d, net.T) for d, net in zip(all_dgms, nets))
  for k, (name, label) in enumerate(MODELS):
    sub = subfigs.ravel()[k]
    net, dgms = nets[k], all_dgms[k]
    density, rate = net.stats()
    sub.suptitle("%s\n" % label + r"$\it{density\ %.2f,\ new\ edge\ rate\ %.2f}$" % (density, rate),
                 fontsize=8, y=1.0)
    gs = sub.add_gridspec(2, 3, height_ratios=[1, 2.1], hspace=0.3, wspace=0.05,
                          left=0.14, right=0.97, top=0.82, bottom=0.12)
    pos = nx.circular_layout(sorted(net.nodes))
    for i, t in enumerate([T / 4, T / 2, 3 * T / 4]):
      draw_snapshot(sub.add_subplot(gs[0, i]), net, t, pos)
    draw_diagram(sub.add_subplot(gs[1, :]), dgms, net.T, legend=(k == 0), pmax=pmax)
  save(fig, "fig4_models")


def fig6_run(args):
  name, seed, n, T, tau = args
  net = make_model(name, n, T, seed)
  dgms = net.flag(max_dim=2).persistence()
  #discrete models end their window with the last step, so use net.T
  return name, seed, net.graph_autocorrelation_continuous(tau), tn.top_autocorr_continuous(dgms, tau, net.T)

def fig6():
  """Topological vs graph autocorrelation across the models."""
  n, T, runs = 40, 40, 30
  tau = T / (T - 1)
  def compute():
    tasks = [(name, seed, n, T, tau) for name, _ in MODELS
             for seed in (range(runs) if name != "sun" else [0])]
    with Pool() as pool:
      return pool.map(fig6_run, tasks)
  rows = pd.DataFrame(cached("fig6", compute), columns=["model", "seed", "graph", "top"])
  fig, ax = plt.subplots(figsize=(0.6 * TEXTWIDTH, 0.5 * TEXTWIDTH))
  handles = []
  for name, label in MODELS:
    r = rows[rows.model == name]
    if name == "sun":
      ax.scatter(r.graph, r.top, marker="*", s=60, color=MODEL_COLORS[name], zorder=3)
      text = label + " (deterministic)"
    else:
      ax.scatter(r.graph, r.top, s=10, alpha=0.75, color=MODEL_COLORS[name], lw=0)
      text = "%s (r = %.2f)" % (label, np.corrcoef(r.graph, r.top)[0, 1])
    handles.append(Line2D([], [], ls="", marker="*" if name == "sun" else "o",
                          color=MODEL_COLORS[name], label=text, markersize=6 if name == "sun" else 4))
  ax.axhline(0, color="0.8", lw=0.6, zorder=0)
  ax.axvline(0, color="0.8", lw=0.6, zorder=0)
  ax.set_xlabel("graph autocorrelation")
  ax.set_ylabel("topological autocorrelation")
  ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
  save(fig, "fig6_autocorrelation")
  print(rows.groupby("model")[["graph", "top"]].agg(["mean", "std"]).round(3))


def hospital_contacts():
  hosp = pd.read_csv(DATA, sep="\t", header=None, names=["t", "i", "j", "s1", "s2"])
  contacts = [(i, j, t / 3600) for (i, j, t) in hosp[["i", "j", "t"]].values.tolist()]
  step = 20 / 3600
  return contacts, step, hosp["t"].max() / 3600 + step

BUFFERS = [(1 / 60, "1 min"), (10 / 60, "10 min"), (1, "1 h"), (4, "4 h"), (10, "10 h"), (24, "24 h"), (48, "48 h")]

def fig11():
  """Hospital data: autocorrelation against buffer, topological autocorrelation against
  lag, and burstiness."""
  contacts, step, T = hospital_contacts()
  tau0 = T / 49
  taus = np.geomspace(0.25, 40, 30).tolist()
  def compute():
    out = []
    for b, label in BUFFERS:
      net = tn.TemporalNetwork.from_contacts(contacts, step + b, T=T)
      dgms = tn.TemporalComplex.from_contacts(contacts, step + b, max_dim=2, T=T).persistence()
      out.append(dict(buffer=label, top=tn.top_autocorr_continuous(dgms, tau0, T),
                      graph=net.graph_autocorrelation_continuous(tau0),
                      top_lag=[tn.top_autocorr_continuous(dgms, tau, T) for tau in taus],
                      burst=[float(x) for x in top_burstiness(dgms, T)]))
    return out
  res = cached("fig11", compute)
  labels = [r["buffer"] for r in res]
  fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.1), gridspec_kw=dict(wspace=0.5))

  ax = axes[0]
  ax.plot(labels, [r["top"] for r in res], marker="o", ms=3, color=DIM_COLORS[2], label="topological")
  ax.plot(labels, [r["graph"] for r in res], marker="s", ms=3, color="0.4", label="graph")
  ax.set_ylim(0, 1)
  ax.set_xlabel("buffer")
  ax.set_ylabel(r"autocorrelation (lag %.2g h)" % tau0)
  ax.tick_params(axis="x", rotation=45)
  ax.legend(frameon=False, loc="lower right")
  ax.set_title("(a)", loc="left")

  ax = axes[1]
  cmap = plt.get_cmap("viridis")
  for k, r in enumerate(res):
    ax.plot(taus, r["top_lag"], color=cmap(k / (len(res) - 1)), lw=1, label=r["buffer"])
  ax.axvline(tau0, color="0.5", lw=0.6, ls=":")
  ax.axvline(24, color="0.5", lw=0.6, ls="--")
  ax.set_xscale("log")
  ax.set_xticks([0.25, 1, 4, 24])
  ax.set_xticklabels(["0.25", "1", "4", "24"])
  ax.set_xlabel("lag (hours)")
  ax.set_ylabel("topological autocorrelation")
  ax.set_title("(b)", loc="left")

  ax = axes[2]
  cols = ["$b_0$", "$d_0$", "$b_1$", "$d_1$", "$b_2$", "$d_2$"]
  for k, r in enumerate(res):
    ax.plot(range(len(r["burst"])), r["burst"], color=cmap(k / (len(res) - 1)), lw=1, marker="o", ms=2,
            label=r["buffer"])
  ax.set_xticks(range(len(cols)))
  ax.set_xticklabels(cols)
  ax.set_xlabel("birth ($b_i$) / death ($d_i$) times in $H_i$")
  ax.set_ylabel("burstiness B")
  ax.set_title("(c)", loc="left")
  ax.legend(title="buffer", frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), handlelength=1.2)
  save(fig, "fig11_hospital")
  for r in res:
    print("%-7s top %.3f graph %.3f" % (r["buffer"], r["top"], r["graph"]))


if __name__ == "__main__":
  which = sys.argv[1:] or ["fig4", "fig6", "fig11"]
  for name in which:
    globals()[name]()
