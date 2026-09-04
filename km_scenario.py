"""Zufaellige 2D-Punktwolken fuer die k-Means-Demo: k Gauss-Cluster, deren Ueberlappung
(spread) und Groessen-/Dichte-Ungleichgewicht (imbalance) unabhaengig einstellbar sind."""

from dataclasses import dataclass

import numpy as np

RING_RADIUS = 3.0
MIN_STD_FRACTION = 0.05  # Untergrenze fuer die Streuung, auch bei sehr kleinem spread


@dataclass(frozen=True)
class ClusteringInstance:
    points: tuple  # ((x, y), ...), Erzeugungsreihenfolge nach Cluster gruppiert
    true_labels: tuple  # welchem erzeugenden Cluster jeder Punkt entstammt
    true_centers: tuple  # ((x, y), ...), k Eintraege
    k: int

    @property
    def n_points(self):
        return len(self.points)

    def as_array(self):
        return np.array(self.points, dtype=float)


def _cluster_shares(k, imbalance):
    """Cluster 0 wird mit wachsendem imbalance groesser (mehr Punkte, mehr Streuung),
    die uebrigen k-1 Cluster entsprechend kleiner und enger - so entsteht bei
    imbalance=1 der Lehrbuchfall 'eine grosse diffuse + mehrere kleine dichte Gruppen',
    bei imbalance=0 sind alle Cluster gleich gross und gleich eng."""
    weights = np.ones(k)
    if k > 1:
        weights[0] = 1 + imbalance * 2 * (k - 1)
    return weights / weights.sum()


def _cluster_stds(k, spread, imbalance):
    base_std = max(spread, MIN_STD_FRACTION) * RING_RADIUS
    stds = np.full(k, base_std)
    if k > 1:
        stds[0] *= 1 + imbalance
        stds[1:] *= max(1 - 0.6 * imbalance, MIN_STD_FRACTION)
    return stds


def generate_instance(n_points, k, spread, imbalance, seed):
    """spread in [0.1, 0.9] steuert die Ueberlappung der Cluster relativ zum
    Ring-Radius (klein = klar getrennt, gross = starke Ueberlappung). imbalance in
    [0, 1] macht Cluster 0 zunehmend groesser und diffuser als die uebrigen - beides
    unabhaengig voneinander regelbar, um Trennbarkeit und Groessen-Ungleichgewicht
    getrennt als Schwierigkeitsachsen zu zeigen."""
    rng = np.random.default_rng(seed)

    angles = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(-0.15, 0.15, size=k)
    true_centers = np.stack([RING_RADIUS * np.cos(angles), RING_RADIUS * np.sin(angles)], axis=1)

    shares = _cluster_shares(k, imbalance)
    stds = _cluster_stds(k, spread, imbalance)
    counts = np.maximum(1, np.round(shares * n_points).astype(int))
    counts[-1] += n_points - counts.sum()  # Rundungsrest auf das letzte Cluster
    counts = np.maximum(counts, 1)

    points_per_cluster = []
    labels_per_cluster = []
    for i in range(k):
        pts = rng.normal(loc=true_centers[i], scale=stds[i], size=(counts[i], 2))
        points_per_cluster.append(pts)
        labels_per_cluster.append(np.full(counts[i], i))

    points = np.concatenate(points_per_cluster, axis=0)
    labels = np.concatenate(labels_per_cluster, axis=0)

    return ClusteringInstance(
        points=tuple(map(tuple, points.tolist())),
        true_labels=tuple(int(l) for l in labels),
        true_centers=tuple(map(tuple, true_centers.tolist())),
        k=k,
    )
