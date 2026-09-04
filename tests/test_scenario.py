import numpy as np

from km_scenario import generate_instance


def test_point_count_matches_request():
    instance = generate_instance(n_points=97, k=4, spread=0.3, imbalance=0.0, seed=1)
    assert instance.n_points == 97
    assert len(instance.true_labels) == 97


def test_reproducible_given_same_seed():
    a = generate_instance(n_points=80, k=3, spread=0.3, imbalance=0.2, seed=42)
    b = generate_instance(n_points=80, k=3, spread=0.3, imbalance=0.2, seed=42)
    assert a.points == b.points
    assert a.true_labels == b.true_labels


def test_different_seed_gives_different_layout():
    a = generate_instance(n_points=80, k=3, spread=0.3, imbalance=0.0, seed=1)
    b = generate_instance(n_points=80, k=3, spread=0.3, imbalance=0.0, seed=2)
    assert a.points != b.points


def test_low_spread_clusters_are_well_separated_from_true_centers():
    """Bei kleinem spread muss jeder Punkt seinem eigenen wahren Zentrum näher sein als
    jedem anderen wahren Zentrum - die klar-getrennte-Gruppen-Prämisse des einfachsten
    Presets muss tatsächlich zutreffen, nicht nur behauptet sein."""
    instance = generate_instance(n_points=60, k=3, spread=0.12, imbalance=0.0, seed=1)
    points = np.array(instance.points)
    true_centers = np.array(instance.true_centers)
    labels = np.array(instance.true_labels)

    d2 = ((points[:, None, :] - true_centers[None, :, :]) ** 2).sum(axis=2)
    nearest_center = d2.argmin(axis=1)
    misassigned_fraction = np.mean(nearest_center != labels)
    assert misassigned_fraction < 0.05


def test_imbalance_makes_cluster_zero_larger():
    instance = generate_instance(n_points=150, k=5, spread=0.3, imbalance=0.9, seed=3)
    labels = np.array(instance.true_labels)
    counts = np.bincount(labels, minlength=5)
    assert counts[0] == counts.max()
    assert counts[0] > counts[1:].mean() * 2
