"""Ausreißer im Szenario: bei 0 bitgleiche Instanz wie vorher, sonst weit außen angehängt, Labels -1, deterministisch."""

import numpy as np
import pytest

import km_scenario as S


def test_zero_outliers_leave_the_instance_unchanged():
    a = S.generate_instance(120, 3, 0.35, 0.0, 7)
    b = S.generate_instance(120, 3, 0.35, 0.0, 7, n_outliers=0)
    assert a == b and a.n_outliers == 0 and a.n_points == 120
    assert a.points[0] == (2.062766219013346, -0.3648448302712489) and a.points[-1] == (-0.49988477438335166, -2.7997150186346786)


@pytest.mark.parametrize("shape", ["blobs", "moons"])
@pytest.mark.parametrize("n_out", [1, 3, 10])
def test_outliers_are_appended_far_away_and_labelled_minus_one(shape, n_out):
    base = S.generate_instance(100, 3, 0.3, 0.0, 5, shape=shape)
    inst = S.generate_instance(100, 3, 0.3, 0.0, 5, shape=shape, n_outliers=n_out)
    assert inst.n_points == 100 + n_out and inst.n_outliers == n_out
    assert inst.points[:100] == base.points and inst.true_labels[:100] == base.true_labels and inst.true_centers == base.true_centers
    assert inst.true_labels[100:] == (-1,) * n_out
    extent = np.linalg.norm(np.array(base.points), axis=1).max()
    norms = np.linalg.norm(np.array(inst.points[100:]), axis=1)
    assert (norms >= S.OUTLIER_MIN_FACTOR * extent - 1e-9).all() and (norms <= S.OUTLIER_MAX_FACTOR * extent + 1e-9).all()


def test_outliers_are_deterministic():
    a = S.generate_instance(60, 3, 0.3, 0.0, 9, n_outliers=5)
    assert a == S.generate_instance(60, 3, 0.3, 0.0, 9, n_outliers=5) and a != S.generate_instance(60, 3, 0.3, 0.0, 10, n_outliers=5)
    assert len(a.points) == 65
