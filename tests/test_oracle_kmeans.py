"""Orakel-Tests (unabhängiger Rechenweg): Lloyd-Schritte gegen eine cdist-Fassung und sklearn, Medoid/Zuordnung/p-Median gegen Brute Force,
Anzahl der erzeugten Punkte gegen den Regler."""

import itertools

import numpy as np
import pytest
from scipy.spatial.distance import cdist

import km_algorithm as A
import km_evaluation as E
import km_exact as X
import km_scenario as S


def _indep_lloyd(data, centers, center, max_iter=50):
    k = len(centers)
    centers = centers.copy()
    d = cdist(data, centers, "sqeuclidean")
    labels = d.argmin(1)
    out = [(centers.copy(), labels.copy(), d[np.arange(len(data)), labels].sum(), len(data))]
    for _ in range(max_iter):
        new = centers.copy()
        for j in range(k):
            idx = np.flatnonzero(labels == j)
            if len(idx):
                new[j] = data[idx].sum(0) / len(idx) if center == "mean" else data[idx[int(np.argmin(cdist(data[idx], data[idx]).sum(0)))]]
        centers = new
        d = cdist(data, centers, "sqeuclidean")
        nl = d.argmin(1)
        ch = int((nl != labels).sum())
        labels = nl
        out.append((centers.copy(), labels.copy(), d[np.arange(len(data)), labels].sum(), ch))
        if ch == 0:
            break
    return out


@pytest.mark.parametrize("center", ["mean", "medoid"])
def test_every_lloyd_step_matches_an_independent_implementation(center):
    rng = np.random.default_rng(1)
    for t in range(25):
        n, k = int(rng.integers(10, 50)), int(rng.integers(2, 6))
        data = S.generate_instance(n, k, 0.3, 0.3, int(rng.integers(10**6)), shape=("blobs", "moons")[t % 2]).as_array()
        res = A.run(data, k, ("random", "kmeans++")[t % 2], int(rng.integers(10**6)), center=center)
        ref = _indep_lloyd(data, np.array(res.steps[0].centers), center)
        assert len(ref) == len(res.steps)
        for s, (c, lab, inertia, ch) in zip(res.steps, ref):
            assert np.allclose(np.array(s.centers), c) and list(s.labels) == list(lab) and s.n_changed == ch
            assert s.inertia == pytest.approx(inertia)


def test_converged_run_matches_sklearn_lloyd_to_the_last_digit():
    cluster = pytest.importorskip("sklearn.cluster")
    for seed in range(10):
        data = S.generate_instance(60, 4, 0.3, 0.4, seed).as_array()
        res = A.run(data, 4, "kmeans++", seed)
        sk = cluster.KMeans(n_clusters=4, init=np.array(res.steps[0].centers), n_init=1, algorithm="lloyd", tol=0, max_iter=300).fit(data)
        assert res.converged
        assert sk.inertia_ == pytest.approx(res.final_inertia, rel=1e-9)
        assert np.allclose(sk.cluster_centers_, np.array(res.final_centers))


def test_duplicate_points_leave_empty_clusters_and_the_objective_never_rises():
    rng = np.random.default_rng(2)
    for t in range(30):
        base = rng.integers(0, 3, size=(3, 2)).astype(float)
        data = base[rng.integers(0, 3, 14)]
        for center in ("mean", "medoid"):
            res = A.run(data, 5, "random", int(rng.integers(10**6)), center=center)
            objs = [s.objective for s in res.steps]
            assert all(b <= a + 1e-9 for a, b in zip(objs, objs[1:]))


def test_kmeans_plusplus_follows_the_d_squared_distribution():
    """Zweites Zentrum bei festem erstem: Wahrscheinlichkeit proportional zum quadrierten Abstand (Häufigkeiten gegen exakte Werte)."""
    data = np.array([[0, 0], [1, 0], [0, 2], [5, 5], [9, 1]], float)
    d2 = cdist(data, data, "sqeuclidean")
    n_runs = 6000
    counts = np.zeros((5, 5))
    for seed in range(n_runs):
        c = A.init_kmeans_plusplus(data, 2, np.random.default_rng(seed))
        i = int(np.flatnonzero((data == c[0]).all(1))[0]); j = int(np.flatnonzero((data == c[1]).all(1))[0])
        counts[i, j] += 1
    for i in range(5):
        row = counts[i] / counts[i].sum()
        assert np.allclose(row, d2[i] / d2[i].sum(), atol=0.04)
    assert np.allclose(counts.sum(1) / n_runs, 0.2, atol=0.02)


def test_center_shift_and_misassigned_share_equal_brute_force_over_all_assignments():
    rng = np.random.default_rng(3)
    for _ in range(60):
        k = int(rng.integers(1, 6))
        a, b = rng.normal(size=(k, 2)), rng.normal(size=(k, 2))
        best = min(np.mean([np.linalg.norm(a[i] - b[p[i]]) for i in range(k)]) for p in itertools.permutations(range(k)))
        assert E.center_shift(a, b) == pytest.approx(best)
        k = max(k, 2)
        n = int(rng.integers(5, 30))
        labels, true = rng.integers(0, k, n), rng.integers(-1, k, n)
        keep = true >= 0
        if not keep.any():
            continue
        hit = max(sum(1 for l, g in zip(labels[keep], true[keep]) if p[l] == g) for p in itertools.permutations(range(k)))
        assert E.misassigned_share(labels, true, k) == pytest.approx(1 - hit / keep.sum())


def test_exact_p_median_equals_brute_force_also_with_duplicate_points():
    rng = np.random.default_rng(4)
    for t in range(12):
        n, k = int(rng.integers(3, 9)), int(rng.integers(1, 4))
        pts = rng.normal(size=(n, 2))
        pts = np.round(pts) if t % 3 == 0 else pts
        d = cdist(pts, pts)
        brute = min(d[:, list(s)].min(1).sum() for s in itertools.combinations(range(n), k))
        value, centers, optimal = X.solve_pmedian(pts, k)
        assert optimal and len(centers) == k and value == pytest.approx(brute, abs=1e-7)


@pytest.mark.parametrize("shape", ["blobs", "moons"])
def test_instance_has_exactly_the_requested_number_of_points_for_small_n_and_many_groups(shape):
    """Regression: bei n = 30..58, k = 7/8 und starkem Ungleichgewicht lieferte der Rundungsrest bis zu zwei Punkte zu viel."""
    for n in (30, 31, 45, 58):
        for k in (7, 8):
            for imbalance in (0.0, 0.4, 0.85, 1.0):
                inst = S.generate_instance(n, k, 0.3, imbalance, 3, shape=shape)
                labels = np.array(inst.true_labels)
                assert len(inst.points) == n and set(labels) == set(range(k))
