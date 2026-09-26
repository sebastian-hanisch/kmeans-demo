"""Exakter p-Median (= k-Medoids-Optimum) als Referenz für die Medoid-Heuristik: die Summe der euklidischen Abstände aller Punkte zum
nächsten von k gewählten Punkten wird per MILP (HiGHS über scipy) minimiert. Nur für kleine Punktzahlen (EXACT_MAX_POINTS), auf Abruf.

Formulierung: y_i = 1, wenn Punkt i Mittelpunkt ist (genau k), x_ji = 1, wenn Punkt j von i bedient wird; x_ji <= y_i. Die Zuordnung ergibt sich bei
ganzzahligem y von selbst (jeder Punkt nimmt den nächsten Mittelpunkt). Wie das Standortproblem der Standortplanungs-Linie (Fixkosten 0, feste Anzahl)."""

from itertools import combinations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def distance_matrix(points):
    pts = np.asarray(points, dtype=float)
    return np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))


def solve_pmedian(points, k, time_limit=30.0):
    """(Kosten, Mittelpunkt-Indizes, optimal?) des exakten p-Medians; bei Zeitlimit die beste gefundene Lösung."""
    d = distance_matrix(points)
    n = len(d)
    nv = n + n * n
    cost = np.concatenate([np.zeros(n), d.ravel()])            # y_i, dann x_ji zeilenweise (j * n + i)
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0
    for j in range(n):
        for i in range(n):
            rows.append(r); cols.append(n + j * n + i); vals.append(1.0)
        lo.append(1.0); hi.append(1.0); r += 1
    for j in range(n):
        for i in range(n):
            rows += [r, r]; cols += [n + j * n + i, i]; vals += [1.0, -1.0]
            lo.append(-np.inf); hi.append(0.0); r += 1
    for i in range(n):
        rows.append(r); cols.append(i); vals.append(1.0)
    lo.append(float(k)); hi.append(float(k)); r += 1
    a = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
    integrality = np.concatenate([np.ones(n), np.zeros(n * n)])
    res = milp(cost, constraints=LinearConstraint(a, np.array(lo), np.array(hi)), integrality=integrality, bounds=Bounds(0, 1),
               options={"time_limit": float(time_limit), "mip_rel_gap": 0.0})
    if res.x is None:
        raise RuntimeError("p-Median nicht lösbar: " + str(res.message))
    centers = tuple(int(i) for i in range(n) if res.x[i] > 0.5)
    return float(d[:, list(centers)].min(axis=1).sum()), centers, res.status == 0


def brute_force(points, k):
    """Alle Auswahlen von k Punkten (nur Kleinstnetze): (Optimum, Liste der optimalen Auswahlen)."""
    d = distance_matrix(points)
    best, sets = None, []
    for s in combinations(range(len(d)), k):
        v = float(d[:, list(s)].min(axis=1).sum())
        if best is None or v < best - 1e-9:
            best, sets = v, [s]
        elif abs(v - best) <= 1e-9:
            sets.append(s)
    return best, sets
