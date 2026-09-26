"""k-Medoids als Zentrum: Medoid-Update von Hand und gegen Brute Force, Monotonie der Summe der Abstände, Konvergenz, Regression des Mittelwert-Modus."""

import numpy as np
import pytest

import km_algorithm as A
import km_scenario as S

ILLUSTRATION = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5], [20.0, 20.0]])


def _instances():
    for n, k, spread, imb, shape, seed in ((120, 3, 0.35, 0.0, "blobs", 7), (150, 5, 0.15, 0.9, "blobs", 1), (150, 2, 0.1, 0.0, "moons", 2), (200, 4, 0.4, 0.3, "blobs", 11)):
        yield S.generate_instance(n, k, spread, imb, seed, shape=shape), k, seed


def test_medoid_by_hand_on_the_illustration_of_the_app():
    """Fünf enge Punkte plus ein Ausreißer bei (20, 20): der Mittelwert liegt bei (3,75; 3,75), der Medoid bleibt der Mittelpunkt (0,5; 0,5)."""
    labels = np.zeros(len(ILLUSTRATION), dtype=int)
    centers = np.array([[9.0, 9.0]])
    assert A._update_centers(ILLUSTRATION, labels, centers, 1)[0] == pytest.approx([3.75, 3.75])
    assert A._update_medoids(ILLUSTRATION, labels, centers, 1)[0] == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize("seed", range(15))
def test_medoid_update_equals_brute_force_and_is_a_data_point(seed):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(40, 2))
    labels = rng.integers(0, 3, size=40)
    centers = np.zeros((3, 2))
    new = A._update_medoids(data, labels, centers, 3)
    for i in range(3):
        pts = data[labels == i]
        if len(pts) == 0:
            assert (new[i] == centers[i]).all()
            continue
        sums = [np.sqrt(((pts - p) ** 2).sum(axis=1)).sum() for p in pts]
        assert any((new[i] == p).all() for p in pts) and np.isclose(min(sums), np.sqrt(((pts - new[i]) ** 2).sum(axis=1)).sum())


def test_medoid_ties_take_the_smallest_index_and_empty_clusters_keep_their_center():
    data = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
    labels = np.array([0, 0, 0])
    centers = np.array([[5.0, 5.0], [7.0, 7.0]])
    new = A._update_medoids(data, labels, centers, 2)
    assert new[0].tolist() == [2.0, 0.0] and new[1].tolist() == [7.0, 7.0]           # Summen 12 / 10 / 18: der mittlere Punkt; Cluster 2 ist leer
    tie = A._update_medoids(np.array([[0.0, 0.0], [2.0, 0.0]]), np.array([0, 0]), np.zeros((1, 2)), 1)
    assert tie[0].tolist() == [0.0, 0.0]                                             # beide Summen 2: der kleinere Index


@pytest.mark.parametrize("init", ["random", "kmeans++"])
def test_the_sum_of_distances_never_increases_and_the_run_converges(init):
    for inst, k, seed in _instances():
        for s in range(3):
            res = A.run(inst.as_array(), k, init, seed + s, center="medoid")
            obj = [st.objective for st in res.steps]
            assert all(b <= a + 1e-9 for a, b in zip(obj, obj[1:]))
            assert res.converged and res.steps[-1].n_changed == 0 and not res.truncated and res.center == "medoid"


def test_every_medoid_center_is_a_data_point_and_objectives_are_consistent():
    inst, k, seed = next(_instances())
    data = inst.as_array()
    res = A.run(data, k, "kmeans++", seed, center="medoid")
    points = {tuple(p) for p in data.tolist()}
    for st in res.steps:
        assert all(tuple(c) in points for c in st.centers)
        centers, labels = np.array(st.centers), np.array(st.labels)
        assert st.total_distance == pytest.approx(np.sqrt(((data - centers[labels]) ** 2).sum(axis=1)).sum()) and st.objective == st.total_distance
    mean_res = A.run(data, k, "kmeans++", seed, center="mean")
    assert all(st.objective == st.inertia for st in mean_res.steps) and mean_res.final_inertia == mean_res.final_objective


def test_mean_mode_is_bit_identical_to_the_version_before_the_switch():
    """Aufgezeichnete Werte des früheren Codes (Lloyd nur mit Mittelwert): Schrittzahl und finale Inertia."""
    expected = [((120, 3, 0.35, 0.0, "blobs", 7, "kmeans++"), 2, 213.57571747503493), ((150, 5, 0.15, 0.9, "blobs", 1, "random"), 16, 170.65913108612617),
                ((150, 2, 0.1, 0.0, "moons", 2, "kmeans++"), 3, 359.0900330943296), ((240, 8, 0.4, 0.3, "blobs", 11, "random"), 8, 346.5072165369352)]
    for (n, k, spread, imb, shape, seed, init), steps, inertia in expected:
        res = A.run(S.generate_instance(n, k, spread, imb, seed, shape=shape).as_array(), k, init, seed)
        assert len(res.steps) == steps and res.final_inertia == pytest.approx(inertia, rel=1e-12) and res.center == "mean"


def test_medoid_and_mean_differ_on_some_instance():
    """Zweig-Test: keine Nullspalte - die beiden Zentren-Arten liefern auf mindestens drei Netzen verschiedene Endzentren."""
    differs = 0
    for inst, k, seed in _instances():
        a = A.run(inst.as_array(), k, "kmeans++", seed, center="mean").final_centers
        b = A.run(inst.as_array(), k, "kmeans++", seed, center="medoid").final_centers
        differs += a != b
    assert differs >= 3


def test_medoid_run_is_deterministic():
    inst, k, seed = next(_instances())
    assert A.run(inst.as_array(), k, "random", seed, center="medoid") == A.run(inst.as_array(), k, "random", seed, center="medoid")


def test_unknown_center_raises():
    inst, k, seed = next(_instances())
    with pytest.raises(KeyError):
        A.run(inst.as_array(), k, "random", seed, center="median")
