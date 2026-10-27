# tempnets

Zigzag persistent homology of temporal networks. `tempnets` builds temporal clique complexes from
random temporal network models or from contact data, computes their zigzag persistence (with
[dionysus](https://mrzv.org/software/dionysus2/)), and summarises the result with autocorrelation
and burstiness statistics.

This code accompanies _[paper reference to be added]_. Background on zigzag persistence and the
interpretation of the statistics is in the paper; this README gives the definitions as implemented.

## Installation

Python 3.10 or later.

```
git clone https://github.com/ggrindstaff/TempNetworks.git
cd TempNetworks
pip install -e .
```

This installs numpy, scipy, pandas, matplotlib, networkx and dionysus. The legacy
`TemporalGraphComplex` class (`tempnets/classes.py`) additionally needs bats-tda, which does not
build on current Python versions; nothing else depends on it.

## Quickstart

```python
import tempnets as tn

# A random model, specified by density and rate of new edges
net = tn.generators.er_periodic(n=40, T=100, density=0.3, rate=0.1, seed=1)
net.stats()                       # NetworkStats(density=0.31, rate=0.11): what was achieved

# Temporal clique complex and its zigzag persistence
cx = net.flag(max_dim=2)
dgms = cx.persistence()           # one array of (birth, death) rows per dimension
tn.plot_diags(dgms, net.T)

# Statistics, with lag tau
tau = net.T / 49
tn.top_autocorr_continuous(dgms, tau, net.T)
net.graph_autocorrelation_continuous(tau)
```

`notebooks/example.ipynb` walks through this on a synthetic model and on the hospital data.

## Definitions

### Temporal networks

A `TemporalNetwork` lives on the time interval [0, T]. Its nodes are present throughout, and each
edge is present on a list of disjoint intervals [start, end) (clipped to [0, T]; overlapping or
touching intervals are merged). Ways to build one:

| Constructor | Input | Convention |
|---|---|---|
| `TemporalNetwork.from_intervals(intervals, T)` | `{(u, v): [(start, end), ...]}` | as given |
| `TemporalNetwork.from_array(arr)` | n × n × T array, `arr[i, j, t] != 0` when i, j are joined at step t | union protocol (below): step t covers [t − 1/2 − ε/2, t + 1/2 + ε/2]; the window ends with the last step, at T − 1/2 + ε/2 |
| `TemporalNetwork.from_contacts(contacts, duration)` | `(u, v, t)` contacts | each contact keeps its edge present on [t, t + duration] |
| `tn.generators.*` | model parameters | see below |

**Discrete time and the union protocol.** Between steps t and t + 1 the complex is the clique
complex of G_t ∪ G_{t+1}, so a feature (for example a cycle that grows by one vertex) can persist
through a change from one step to the next: the zigzag is G_t ↪ H_t ↩ G_{t+1}. This is implemented
by letting consecutive steps overlap for a short time ε (`overlap`, default 0.001), during which
both graphs, and any cliques they form together, are present. Births and deaths are at half-integer
times, shifted by at most ε/2. Features that exist only in some H_t appear with persistence ε.

`net.level_set(t)` is the graph at time t, `net.to_array(times)` samples the network, and
`net.stats()` reports its density and rate.

### Density and rate

Every random model is specified by

- **density**: the expected fraction of the n(n − 1)/2 node pairs joined by an edge at a given time;
- **rate**: the expected number of edge appearances (absent → present) per node pair per unit time,
  so `rate · n(n − 1)/2` new edges per unit time across the network.

In a stationary network these fix the mean lifetime of an edge at density/rate. Each model's own
parameters are computed from them (`tempnets/params.py`):

| Generator | Model | Parameters from density and rate |
|---|---|---|
| `binomial(n, T, density, rate)` | discrete time; each pair is a two-state Markov chain | P(absent → present) = rate/(1 − density), P(present → absent) = rate/density; started at stationarity. Needs rate ≤ min(density, 1 − density). |
| `poisson(n, T, density)` | discrete time; one-step contacts separated by Poisson(λ) gaps, started at stationarity | λ solves (1 − e^(−λ))/λ = density. Contacts last one step, so the rate cannot be chosen. |
| `er_periodic(n, T, density, rate, p=None)` | each edge of G(n, p) is present for density/rate time units roughly every p/rate | p is free with density < p ≤ 1; default p = min(1, 2·density), so edges are present about half the time |
| `ws_rewiring(n, T, density, rate)` | ring lattice whose edges are rewired one at a time at exponential intervals | neighbour count k = density·(n − 1) rounded to an even number (so the density is matched as closely as k allows); mean time between rewirings 1/(rate · n(n − 1)/2) |
| `topological_rewiring(n, T, density, rate)` | starts from G(n, density); each rewiring removes an edge lying on a cycle (if any) and adds a random edge | same rewiring interval as above. Suppresses homology only in sparse networks (average degree about 2): at higher density most graph cycles are filled by triangles |
| `sun(n, T, rays=5)` | deterministic structured network | not matched to a density or rate |

All generators take `seed=` for reproducible networks.

### Temporal clique complexes

- `net.flag(max_dim)`: every clique of up to `max_dim + 1` nodes is present on the intervals where
  all of its edges are present.
- `TemporalComplex.from_contacts(contacts, duration, max_dim)`: for contact data, a clique is present
  only at the recorded times when **all of its pairs are in contact simultaneously**, and each such
  time keeps it present for `duration` (as `from_contacts` does for edges). This is the rule used for
  the hospital data; it gives fewer higher simplices than `from_contacts(...).flag()`, where any
  overlap of the buffered edges forms a clique.

`cx.persistence()` returns the zigzag persistence diagrams as one array of (birth, death) rows per
dimension. Classes alive at the end of the window die at T and are drawn on the ∞ line by
`plot_diags`. (dionysus stores times as 32-bit floats; intervals shorter than that resolution are
dropped before the computation, and reported times are mapped back to the exact event times.)

### Statistics

The autocorrelation statistics measure how much of what is present at a time t is still present a
lag τ later, and how persistent that similarity is over time. For t in [0, T − τ],

- topological: γ(t) = |A(t) ∩ A(t + τ)| / √(|A(t)| · |A(t + τ)|), where A(t) is the set of diagram
  points alive at t (0 if either set is empty);
- graph: γ(t) is the same cosine similarity between each node's incident edges at t and at t + τ,
  averaged over nodes;

and the statistic is the autocorrelation of γ at lag τ:

ρ = ∫₀^(T−2τ) (γ(t) − γ̄)(γ(t + τ) − γ̄) dt / ∫₀^(T−τ) (γ(t) − γ̄)² dt,  with γ̄ the mean of γ on [0, T − τ].

These are `tn.top_autocorr_continuous(dgms, tau, T)` and
`net.graph_autocorrelation_continuous(tau)`. γ is a step function, so the integrals are computed
exactly; no sampling grid is involved. The analyses use τ = T/49 (hospital) and T/(T − 1) (models),
the spacing of the snapshot grids used previously. The snapshot versions, which evaluate γ only on
a grid of snapshots spaced τ apart, remain available as `top_autocorr` and
`net.graph_autocorrelation`.

Burstiness is measured with the coefficient B = (σ − μ)/(σ + μ) of a set of waiting times
(−1 periodic, 0 Poisson-like, towards 1 bursty); in the code this is `topfunctions.cv`, used by
`burstiness` (inter-contact times of edges), `pers_burstiness` (persistences) and `top_burstiness`
(gaps between birth times and between death times, per dimension).

## Reproducing the figures

From the repository root:

```
python tests/hospital.py          # hospital diagrams and summaries, about 15 s
python -m tempnets.models         # random-model diagrams and the ER experiment, about 15 min
```

Figures are written to `figs/`.

## Data

`data/detailed_list_of_contacts_Hospital.dat` is the hospital ward contact data collected by the
[SocioPatterns](http://www.sociopatterns.org) collaboration; see their site for terms of use. Each
line is `t i j Si Sj`: the time in seconds (20-second resolution), the two individuals in contact,
and their roles (ADM administrative staff, MED medical doctors, NUR nurses and nurses' aides, PAT
patients).

> P. Vanhems, A. Barrat, C. Cattuto, J.-F. Pinton, N. Khanafer, C. Régis, B. Kim, B. Comte and
> N. Voirin. Estimating potential infection transmission routes in hospital wards using wearable
> proximity sensors. *PLoS ONE* 8(9): e73970 (2013).
> [doi:10.1371/journal.pone.0073970](https://doi.org/10.1371/journal.pone.0073970)

## Package layout

| Module | Contents |
|---|---|
| `tempnets/network.py` | `TemporalNetwork`, `TemporalComplex` |
| `tempnets/generators.py` | random models specified by density and rate |
| `tempnets/params.py` | model parameters from density and rate; `interval_stats`, `array_stats` |
| `tempnets/topfunctions.py` | autocorrelation and burstiness statistics |
| `tempnets/utils.py` | lower-level building blocks: interval conversions, models with explicit parameters, zigzag persistence, plotting |
| `tempnets/discrete.py` | discrete-time (n × n × T array) generators and helpers |
| `tempnets/models.py` | the random-model experiments (a script) |
| `tests/hospital.py` | the hospital analysis (a script) |
| `tests/test_*.py` | unit tests: `pytest` |

## Citation

If you use this code, please cite _[paper reference to be added]_.
