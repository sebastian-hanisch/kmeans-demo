"""Exakter p-Median: MILP gegen Brute Force auf Kleinstnetzen, Randfälle."""

import numpy as np
import pytest

import km_algorithm as A
import km_exact as X
import km_scenario as S


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("k", [1, 2, 3])
def test_milp_equals_brute_force(seed, k):
    inst = S.generate_instance(12, 2, 0.4, 0.0, seed)
    opt, centers, proven = X.solve_pmedian(inst.points, k)
    best, sets = X.brute_force(inst.points, k)
    assert proven and opt == pytest.approx(best, abs=1e-7) and len(centers) == k and tuple(sorted(centers)) in sets


def test_k_equals_n_costs_nothing_and_k_one_is_the_best_single_point():
    pts = [(0.0, 0.0), (1.0, 0.0), (5.0, 0.0)]
    assert X.solve_pmedian(pts, 3)[0] == pytest.approx(0.0)
    opt, centers, _ = X.solve_pmedian(pts, 1)
    assert centers == (1,) and opt == pytest.approx(1.0 + 4.0)


def test_exact_is_never_worse_than_the_heuristic():
    for seed in range(5):
        inst = S.generate_instance(40, 3, 0.35, 0.0, seed)
        opt, _c, _p = X.solve_pmedian(inst.points, 3)
        assert A.run(inst.as_array(), 3, "kmeans++", seed, center="medoid").final_total_distance >= opt - 1e-7


def test_distance_matrix_is_symmetric():
    d = X.distance_matrix(S.generate_instance(10, 2, 0.4, 0.0, 1).points)
    assert np.allclose(d, d.T) and np.allclose(np.diag(d), 0)
