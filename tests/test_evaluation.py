from km_evaluation import multistart_comparison
from km_scenario import generate_instance


def test_global_best_is_the_minimum_of_both_strategies():
    instance = generate_instance(n_points=90, k=4, spread=0.35, imbalance=0.3, seed=5)
    comparison = multistart_comparison(instance.as_array(), k=4, n_restarts=20, base_seed=5)
    best_random = comparison.strategies["random"].best_inertia
    best_kpp = comparison.strategies["kmeans++"].best_inertia
    assert comparison.global_best_inertia == min(best_random, best_kpp)


def test_near_best_fraction_is_a_valid_share():
    instance = generate_instance(n_points=90, k=4, spread=0.35, imbalance=0.3, seed=5)
    comparison = multistart_comparison(instance.as_array(), k=4, n_restarts=20, base_seed=5)
    for summary in comparison.strategies.values():
        assert 0.0 <= summary.near_best_fraction <= 1.0
        assert len(summary.final_inertias) == 20
        assert summary.best_inertia == min(summary.final_inertias)


def test_easy_well_separated_scenario_favors_kmeanspp_almost_always():
    """Bei klar getrennten, gleich grossen Gruppen (das einfachste Preset) findet
    k-Means++ so gut wie immer das beste Ergebnis - die D^2-Gewichtung trifft praktisch
    immer je einen Startpunkt pro Gruppe. Reine Zufalls-Init dagegen kann selbst hier
    noch danebengreifen (zwei Startpunkte in derselben Gruppe, eine Gruppe bleibt ohne
    eigenes Zentrum) - genau der Punkt, den die '📐'-Sektion der App live zeigt: schon im
    einfachsten Fall ist Zufalls-Init nicht risikofrei, k-Means++ schon."""
    instance = generate_instance(n_points=60, k=3, spread=0.15, imbalance=0.0, seed=1)
    comparison = multistart_comparison(instance.as_array(), k=3, n_restarts=30, base_seed=1)
    assert comparison.strategies["kmeans++"].near_best_fraction > 0.9
    assert comparison.strategies["kmeans++"].near_best_fraction >= comparison.strategies["random"].near_best_fraction


def test_hard_imbalanced_scenario_shows_a_clear_gap():
    """Beim 'schweren Fall'-Preset (ungleiche Gruppengroessen) soll k-Means++ im
    Mittel spuerbar bessere (kleinere) finale Inertia erreichen als Zufalls-Init -
    genau die Behauptung, die die App live in der '📐'-Sektion nachweist."""
    instance = generate_instance(n_points=150, k=5, spread=0.15, imbalance=0.9, seed=1)
    comparison = multistart_comparison(instance.as_array(), k=5, n_restarts=40, base_seed=1)
    random_summary = comparison.strategies["random"]
    kpp_summary = comparison.strategies["kmeans++"]
    assert kpp_summary.mean_inertia <= random_summary.mean_inertia
    assert kpp_summary.near_best_fraction >= random_summary.near_best_fraction
