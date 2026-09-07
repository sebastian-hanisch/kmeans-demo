"""Zufällige 2D-Punktwolken für die k-Means-Demo: k Gauss-Cluster ("blobs") oder k
nicht-konvexe Halbkreis-Bögen ("moons", wie in dbscan-demo/spectral-demo generalisiert -
bei k=2 das klassische Zwei-Halbmonde-Beispiel, bei k>2 k Bögen wie Blütenblätter auf
einem Ring), deren Überlappung/Rauschen (spread) und Größen-/Dichte-Ungleichgewicht
(imbalance) bei BEIDEN Formen unabhängig einstellbar sind."""

from dataclasses import dataclass

import numpy as np

RING_RADIUS = 3.0
ARC_RADIUS = 2.5
ARC_RING_RADIUS = 6.5
MIN_STD_FRACTION = 0.05  # Untergrenze für die Streuung, auch bei sehr kleinem spread


@dataclass(frozen=True)
class ClusteringInstance:
    points: tuple  # ((x, y), ...), Erzeugungsreihenfolge nach Cluster gruppiert
    true_labels: tuple  # welchem erzeugenden Cluster jeder Punkt entstammt
    true_centers: tuple  # ((x, y), ...), k Einträge - bei "moons" der Schwerpunkt je Bogen
    shape: str  # "blobs" oder "moons"
    k: int

    @property
    def n_points(self):
        return len(self.points)

    def as_array(self):
        return np.array(self.points, dtype=float)


def _cluster_shares(k, imbalance):
    """Cluster 0 wird mit wachsendem imbalance größer (mehr Punkte, mehr Streuung),
    die übrigen k-1 Cluster entsprechend kleiner und enger - so entsteht bei
    imbalance=1 der Lehrbuchfall 'eine große diffuse + mehrere kleine dichte Gruppen',
    bei imbalance=0 sind alle Cluster gleich groß und gleich eng."""
    weights = np.ones(k)
    if k > 1:
        weights[0] = 1 + imbalance * 2 * (k - 1)
    return weights / weights.sum()


def _cluster_stds(k, imbalance, base_std):
    stds = np.full(k, base_std)
    if k > 1:
        stds[0] *= 1 + imbalance
        stds[1:] *= max(1 - 0.6 * imbalance, MIN_STD_FRACTION)
    return stds


def _counts_from_shares(k, imbalance, n_points):
    shares = _cluster_shares(k, imbalance)
    counts = np.maximum(1, np.round(shares * n_points).astype(int))
    counts[-1] += n_points - counts.sum()  # Rundungsrest auf das letzte Cluster
    return np.maximum(counts, 1)


def _generate_blobs(n_points, k, spread, imbalance, rng):
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(-0.15, 0.15, size=k)
    true_centers = np.stack([RING_RADIUS * np.cos(angles), RING_RADIUS * np.sin(angles)], axis=1)

    counts = _counts_from_shares(k, imbalance, n_points)
    base_std = max(spread, MIN_STD_FRACTION) * RING_RADIUS
    stds = _cluster_stds(k, imbalance, base_std)

    points_per_cluster, labels_per_cluster = [], []
    for i in range(k):
        pts = rng.normal(loc=true_centers[i], scale=stds[i], size=(counts[i], 2))
        points_per_cluster.append(pts)
        labels_per_cluster.append(np.full(counts[i], i))

    points = np.concatenate(points_per_cluster, axis=0)
    labels = np.concatenate(labels_per_cluster, axis=0)
    return points, labels, true_centers


def _generate_moons(n_points, k, spread, imbalance, rng):
    """k=2: das klassische "two moons"-Beispiel (wie in dbscan-demo/spectral-demo).
    k>2: k Halbkreis-Bögen wie Blütenblätter auf einem Ring, konkave Seite zum Zentrum.
    imbalance wirkt hier wie bei "blobs": Gruppe 0 bekommt mehr Punkte UND mehr Rauschen,
    die übrigen entsprechend weniger/enger - dieselbe Größen-UND-Dichte-Kopplung, nur auf
    die Bogenform übertragen statt auf runde Cluster."""
    counts = _counts_from_shares(k, imbalance, n_points)
    base_noise_std = max(spread, MIN_STD_FRACTION) * ARC_RADIUS * 0.3
    noise_stds = _cluster_stds(k, imbalance, base_noise_std)

    if k == 2:
        t1 = rng.uniform(0, np.pi, counts[0])
        x1 = ARC_RADIUS * np.cos(t1)
        y1 = ARC_RADIUS * np.sin(t1)

        t2 = rng.uniform(0, np.pi, counts[1])
        x2 = ARC_RADIUS * (1 - np.cos(t2))
        y2 = ARC_RADIUS * (0.5 - np.sin(t2))

        pts1 = np.stack([x1, y1], axis=1) + rng.normal(scale=noise_stds[0], size=(counts[0], 2))
        pts2 = np.stack([x2, y2], axis=1) + rng.normal(scale=noise_stds[1], size=(counts[1], 2))
        points = np.concatenate([pts1, pts2], axis=0)
        labels = np.concatenate([np.zeros(counts[0], dtype=int), np.ones(counts[1], dtype=int)])
        true_centers = np.stack([points[labels == i].mean(axis=0) for i in range(k)])
        return points, labels, true_centers

    layout_angles = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(-0.1, 0.1, size=k)
    arc_centers = np.stack(
        [ARC_RING_RADIUS * np.cos(layout_angles), ARC_RING_RADIUS * np.sin(layout_angles)], axis=1
    )

    points_per_group, labels_per_group = [], []
    for i in range(k):
        t = rng.uniform(0, np.pi, counts[i])
        local_x = ARC_RADIUS * np.cos(t)
        local_y = ARC_RADIUS * np.sin(t)
        # Um layout_angle_i + pi rotieren, damit die konkave Seite des Bogens zum
        # Ringzentrum zeigt (Blütenblatt-Anordnung), statt nach außen.
        rot = layout_angles[i] + np.pi
        cos_r, sin_r = np.cos(rot), np.sin(rot)
        rx = cos_r * local_x - sin_r * local_y
        ry = sin_r * local_x + cos_r * local_y
        pts = np.stack([rx, ry], axis=1) + arc_centers[i] + rng.normal(scale=noise_stds[i], size=(counts[i], 2))
        points_per_group.append(pts)
        labels_per_group.append(np.full(counts[i], i))

    points = np.concatenate(points_per_group, axis=0)
    labels = np.concatenate(labels_per_group, axis=0)
    true_centers = np.stack([points[labels == i].mean(axis=0) for i in range(k)])
    return points, labels, true_centers


def generate_instance(n_points, k, spread, imbalance, seed, shape="blobs"):
    """spread in [0.1, 0.9] steuert je nach shape entweder die Überlappung der Cluster
    relativ zum Ring-Radius ("blobs") oder das Rauschen um die ideale Bogen-Kurve
    ("moons"). imbalance in [0, 1] macht Gruppe 0 zunehmend größer und diffuser als die
    übrigen - bei BEIDEN Formen, unabhängig von der Trennbarkeit regelbar."""
    rng = np.random.default_rng(seed)

    if shape == "moons":
        points, labels, true_centers = _generate_moons(n_points, k, spread, imbalance, rng)
    else:
        points, labels, true_centers = _generate_blobs(n_points, k, spread, imbalance, rng)

    return ClusteringInstance(
        points=tuple(map(tuple, points.tolist())),
        true_labels=tuple(int(l) for l in labels),
        true_centers=tuple(map(tuple, true_centers.tolist())),
        shape=shape,
        k=k,
    )
