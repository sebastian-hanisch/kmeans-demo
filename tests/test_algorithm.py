import numpy as np
import pytest

from km_algorithm import _assign, _update_centers, init_kmeans_plusplus, init_random, run
from km_scenario import generate_instance


def test_inertia_never_increases_random_init():
    for seed in range(15):
        instance = generate_instance(n_points=80, k=4, spread=0.4, imbalance=0.3, seed=seed)
        result = run(instance.as_array(), k=4, init_strategy="random", seed=seed)
        inertias = [s.inertia for s in result.steps]
        for before, after in zip(inertias, inertias[1:]):
            assert after <= before + 1e-9


def test_inertia_never_increases_kmeanspp_init():
    for seed in range(15):
        instance = generate_instance(n_points=80, k=4, spread=0.4, imbalance=0.3, seed=seed)
        result = run(instance.as_array(), k=4, init_strategy="kmeans++", seed=seed)
        inertias = [s.inertia for s in result.steps]
        for before, after in zip(inertias, inertias[1:]):
            assert after <= before + 1e-9


def test_converges_within_max_iter_for_typical_scenarios():
    for seed in range(10):
        instance = generate_instance(n_points=100, k=5, spread=0.3, imbalance=0.2, seed=seed)
        result = run(instance.as_array(), k=5, init_strategy="kmeans++", seed=seed)
        assert result.converged
        assert not result.truncated


def test_last_step_has_zero_changed_points_when_converged():
    instance = generate_instance(n_points=90, k=3, spread=0.3, imbalance=0.0, seed=2)
    result = run(instance.as_array(), k=3, init_strategy="random", seed=2)
    assert result.converged
    assert result.final_step.n_changed == 0


def test_well_placed_init_converges_immediately_to_the_obvious_partition():
    """Zwei weit getrennte Punktpaare, Startzentren genau in der Mitte jedes Paares -
    das ist bereits die optimale Partition, ein Update-Schritt darf daran nichts mehr
    ändern (deterministischer Nachweis der Kernrechnung: Zuweisung + Schwerpunkt-Update)."""
    data = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 0.0], [10.0, 1.0]])
    centers = np.array([[0.0, 0.5], [10.0, 0.5]])

    labels, inertia = _assign(data, centers)
    assert list(labels) == [0, 0, 1, 1]
    assert inertia == pytest.approx(1.0)

    updated = _update_centers(data, labels, centers, k=2)
    assert updated == pytest.approx(centers)


def test_bad_init_can_converge_to_a_worse_local_optimum():
    """Dieselben vier Punkte, aber mit beiden Startzentren im selben (linken) Paar -
    Lloyd's Algorithmus konvergiert stabil, aber auf eine schlechtere Partition
    (nach x statt nach den tatsächlichen Paaren getrennt). Das ist der Kern-Beleg
    dafür, dass Konvergenz allein keine Optimalität garantiert - die Existenzgrundlage
    der gesamten Demo, hier exakt nachgerechnet statt nur behauptet."""
    data = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 0.0], [10.0, 1.0]])
    bad_centers = np.array([[0.0, 0.0], [0.0, 1.0]])

    centers = bad_centers
    labels, inertia = _assign(data, centers)
    for _ in range(10):
        centers = _update_centers(data, labels, centers, k=2)
        new_labels, new_inertia = _assign(data, centers)
        if list(new_labels) == list(labels):
            labels, inertia = new_labels, new_inertia
            break
        labels, inertia = new_labels, new_inertia

    assert inertia == pytest.approx(100.0)
    assert inertia > 1.0  # schlechter als die optimale Partition aus dem Test oben


def test_init_random_returns_k_distinct_data_points():
    data = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    rng = np.random.default_rng(0)
    centers = init_random(data, k=3, rng=rng)
    assert centers.shape == (3, 2)
    for c in centers:
        assert any(np.allclose(c, p) for p in data)


def test_init_kmeans_plusplus_favors_the_far_small_cluster():
    """Statistische Absicherung des D^2-Gewichtungsschemas: bei einer stark
    unausgewogenen Instanz (ein großes Cluster nahe dem Ursprung, ein kleines, weit
    entferntes) muss k-Means++ das kleine Cluster deutlich häufiger als Startzentrum
    treffen als eine gleichverteilte Zufallsauswahl - sonst wäre die Implementierung
    keine echte D^2-Gewichtung, sondern liefe faktisch auf Uniform-Sampling hinaus."""
    big_cluster = np.random.default_rng(1).normal(loc=[0, 0], scale=0.2, size=(95, 2))
    small_cluster = np.random.default_rng(2).normal(loc=[50, 50], scale=0.2, size=(5, 2))
    data = np.concatenate([big_cluster, small_cluster], axis=0)

    hits_kpp = 0
    hits_uniform = 0
    n_trials = 300
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        centers = init_kmeans_plusplus(data, k=2, rng=rng)
        if any(np.linalg.norm(c - [50, 50]) < 5 for c in centers):
            hits_kpp += 1

        rng2 = np.random.default_rng(seed)
        idx = rng2.choice(len(data), size=2, replace=False)
        if any(np.linalg.norm(data[i] - [50, 50]) < 5 for i in idx):
            hits_uniform += 1

    assert hits_kpp / n_trials > hits_uniform / n_trials
    assert hits_kpp / n_trials > 0.85


def test_matches_sklearn_kmeans_from_the_same_initial_centers():
    """Unabhängiger Kreuzvergleich der eigenen Lloyd-Implementierung gegen
    sklearn.cluster.KMeans: von denselben Startzentren aus muss dieselbe (oder eine
    inertia-gleichwertige) Konvergenz erreicht werden. sklearn ist ausschließlich ein
    Test-Dependency, kein Laufzeit-Dependency der App."""
    sklearn = pytest.importorskip("sklearn.cluster")

    for seed in range(8):
        instance = generate_instance(n_points=70, k=3, spread=0.35, imbalance=0.4, seed=seed)
        data = instance.as_array()
        rng = np.random.default_rng(seed)
        start_centers = init_kmeans_plusplus(data, k=3, rng=rng)

        centers, labels = start_centers.copy(), None
        for _ in range(50):
            new_labels, new_inertia = _assign(data, centers)
            if labels is not None and list(new_labels) == list(labels):
                break
            labels = new_labels
            centers = _update_centers(data, labels, centers, k=3)
        _, our_inertia = _assign(data, centers)

        sk_model = sklearn.KMeans(n_clusters=3, init=start_centers.copy(), n_init=1, max_iter=50)
        sk_model.fit(data)

        assert our_inertia == pytest.approx(sk_model.inertia_, rel=1e-3)
