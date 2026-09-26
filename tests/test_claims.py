"""Jede neue Zahl zum Mittelwert-oder-Medoid-Schalter, die README und App nennen, ist hier belegt: Preset-Instanz (120 Punkte, 3 Gruppen, 5 Ausreißer, Seed 3) und Verteilungen über 40 feste Netze
(Seeds ab 100000). Werte der Lösungen sind Gleitkommazahlen (Vergleich mit Toleranz); gezählt werden nur Größen, die nicht davon abhängen, welche von mehreren gleich guten Auswahlen gewählt wird."""

import statistics

import pytest

import km_algorithm as A
import km_constants as C
import km_evaluation as E
import km_exact as X
import km_scenario as S

PCT = pytest.approx


def _preset_instances():
    clean = S.generate_instance(120, 3, 0.25, 0.0, 3, "blobs", 0)
    dirty = S.generate_instance(120, 3, 0.25, 0.0, 3, "blobs", 5)
    return clean, dirty


@pytest.fixture(scope="module")
def experiment():
    return E.outlier_experiment(120, 3, 0.25, 0.0, "blobs")


def test_preset_instance_outliers_pull_the_mean_but_not_the_medoid():
    """Seed 3, 5 Ausreißer, bestes von 10 Läufen: Mittelwert Verschiebung der Zentren 4,87 und 33,3 % falsch zugeordnet (eine echte Gruppe wird an die Ausreißer abgegeben); Medoid Verschiebung 0,00 und 0 %."""
    clean, dirty = _preset_instances()
    mean = E.robustness(clean, dirty, 3, "mean", C.N_RESTARTS_COMPARE, C.COMPARE_SEED)
    medoid = E.robustness(clean, dirty, 3, "medoid", C.N_RESTARTS_COMPARE, C.COMPARE_SEED)
    assert (mean["shift"], mean["misassigned"]) == (PCT(4.874, abs=0.001), PCT(1 / 3, abs=1e-9))
    assert (medoid["shift"], medoid["misassigned"]) == (PCT(0.0, abs=1e-9), 0.0)


def test_preset_instance_each_mode_wins_in_its_own_measure():
    """Bestes von 10 Läufen: Mittelwert Inertia 873,7 und Summe der Abstände 273,6; Medoid Inertia 1 004,5 und Summe der Abstände 176,3."""
    _clean, dirty = _preset_instances()
    cmp = E.compare_centers(dirty.as_array(), 3, C.N_RESTARTS_COMPARE, C.COMPARE_SEED)
    assert (cmp["mean"].final_step.inertia, cmp["mean"].final_step.total_distance) == (PCT(873.7, abs=0.05), PCT(273.6, abs=0.05))
    assert (cmp["medoid"].final_step.inertia, cmp["medoid"].final_step.total_distance) == (PCT(1004.5, abs=0.05), PCT(176.3, abs=0.05))


def test_preset_instance_multistart_with_outliers():
    """Medoid, 40 Läufe je Start-Strategie: Ø Summe der Abstände 198,7 bei Zufalls-Start gegen 254,6 bei k-Means++ (nahe am Besten 176,3: 80 % gegen 35 %) - k-Means++ wählt die weit entfernten Ausreißer bevorzugt als Startpunkte."""
    _clean, dirty = _preset_instances()
    comp = E.multistart_comparison(dirty.as_array(), 3, C.N_RESTARTS_MULTISTART, C.COMPARISON_SEED, "medoid")
    assert comp.strategies["random"].mean_inertia == PCT(198.7, abs=0.05) and comp.strategies["kmeans++"].mean_inertia == PCT(254.6, abs=0.05) and comp.global_best_inertia == PCT(176.3, abs=0.05)
    assert (comp.strategies["random"].near_best_fraction, comp.strategies["kmeans++"].near_best_fraction) == (PCT(0.8), PCT(0.35))


def test_experiment_shift_by_number_of_outliers(experiment):
    """40 Netze (120 Punkte, 3 Gruppen, Streuung 0,25), 1 / 3 / 5 / 10 Ausreißer: mittlere Verschiebung der Zentren Mittelwert 0,106 / 0,578 / 0,836 / 3,573, Medoid 0,011 / 0,050 / 0,075 / 2,056;
    Median Mittelwert 0,106 / 0,287 / 0,469 / 4,689, Medoid 0,000 / 0,000 / 0,014 / 0,220; der Medoid wird in 37 / 40 / 40 / 33 von 40 Netzen weniger verschoben."""
    ns = (1, 3, 5, 10)
    assert [experiment[n]["shift_mean"]["mean"] for n in ns] == PCT([0.1057, 0.5777, 0.8359, 3.5725], abs=0.001)
    assert [experiment[n]["shift_mean"]["medoid"] for n in ns] == PCT([0.0108, 0.0501, 0.0749, 2.0558], abs=0.001)
    assert [experiment[n]["shift_median"]["mean"] for n in ns] == PCT([0.1056, 0.2869, 0.4693, 4.689], abs=0.001)
    assert [experiment[n]["shift_median"]["medoid"] for n in ns] == PCT([0.0, 0.0, 0.0143, 0.2195], abs=0.001)
    assert [experiment[n]["medoid_smaller"] for n in ns] == [37, 40, 40, 33] and all(experiment[n]["count"] == 40 for n in ns)


def test_experiment_misassignment_by_number_of_outliers(experiment):
    """Falsch zugeordnete echte Punkte im Mittel: Mittelwert 0,1 / 1,8 / 2,7 / 21,8 %, Medoid 0,1 / 0,1 / 0,1 / 12,6 %; schlechtestes Netz bei 3 Ausreißern 33,3 % gegen 0,8 %, bei 10 Ausreißern 66,7 % gegen 34,2 %."""
    ns = (1, 3, 5, 10)
    assert [100 * experiment[n]["mis_mean"]["mean"] for n in ns] == PCT([0.10, 1.79, 2.67, 21.79], abs=0.01)
    assert [100 * experiment[n]["mis_mean"]["medoid"] for n in ns] == PCT([0.12, 0.10, 0.10, 12.60], abs=0.01)
    assert (100 * experiment[3]["mis_max"]["mean"], 100 * experiment[3]["mis_max"]["medoid"]) == (PCT(33.33, abs=0.01), PCT(0.83, abs=0.01))
    assert (100 * experiment[10]["mis_max"]["mean"], 100 * experiment[10]["mis_max"]["medoid"]) == (PCT(66.67, abs=0.01), PCT(34.17, abs=0.01))


def test_clean_data_cost_of_the_medoid():
    """Ohne Ausreißer (40 Netze, bestes von 10 Läufen): die Medoid-Lösung hat im Mittel die 1,031-fache Inertia des Mittelwerts (höchstens 1,081, in 40 von 40 Netzen größer); der Mittelwert hat die 0,9955-fache
    Summe der Abstände des Medoids (0,987 bis 1,004, höchstens 1 in 33 Netzen); beide brauchen im Mittel 1,4 Iterationen (1,375 gegen 1,425)."""
    rq, rd, im, idd = [], [], [], []
    for seed in C.EXPERIMENT_SEEDS:
        d = S.generate_instance(120, 3, 0.25, 0.0, seed, "blobs", 0).as_array()
        res = E.compare_centers(d, 3, C.N_RESTARTS_COMPARE, C.COMPARE_SEED)
        rq.append(res["medoid"].final_step.inertia / res["mean"].final_step.inertia)
        rd.append(res["mean"].final_step.total_distance / res["medoid"].final_step.total_distance)
        im.append(len(res["mean"].steps) - 1)
        idd.append(len(res["medoid"].steps) - 1)
    assert statistics.fmean(rq) == PCT(1.0313, abs=0.0005) and max(rq) == PCT(1.0807, abs=0.0005) and all(x > 1 for x in rq)
    assert statistics.fmean(rd) == PCT(0.9955, abs=0.0005) and (min(rd), max(rd)) == (PCT(0.9868, abs=0.0005), PCT(1.004, abs=0.0005)) and sum(1 for x in rd if x <= 1 + 1e-9) == 33
    assert (statistics.fmean(im), statistics.fmean(idd)) == (PCT(1.375, abs=0.001), PCT(1.425, abs=0.001))


@pytest.mark.parametrize("k, single_kpp, single_rnd, best, best_max, exact, kmeans, kmeans_min, kmeans_max", [
    (3, 1.0239, 1.1173, 1.0000, 1.0000, 40, 0.9975, 0.976, 1.025),
    (5, 1.0806, 1.0974, 1.0002, 1.006, 39, 1.0056, 0.981, 1.025),
])
def test_medoid_heuristic_against_the_exact_p_median(k, single_kpp, single_rnd, best, best_max, exact, kmeans, kmeans_min, kmeans_max):
    """60 Punkte, 3 bzw. 5 Gruppen (Streuung 0,35), 40 Netze: ein k-Medoids-Lauf mit k-Means++-Start ist im Mittel das 1,024- bzw. 1,081-Fache des exakten p-Medians, mit Zufalls-Start 1,117 bzw. 1,097; das beste von
    40 Läufen trifft das Optimum in 40 bzw. 39 von 40 Netzen (im Mittel 1,0000 bzw. 1,0002); die k-Means-Lösung (Mittelwert-Zentren, bestes von 40) hat die Summe der Abstände 0,998 bzw. 1,006 des p-Median-Optimums (0,976 bis 1,025 bzw. 0,981 bis 1,025)."""
    g = {"kpp": [], "rnd": [], "best": [], "kmeans": []}
    n_exact = 0
    for seed in C.EXPERIMENT_SEEDS:
        inst = S.generate_instance(60, k, 0.35, 0.0, seed, "blobs", 0)
        d = inst.as_array()
        opt, _c, proven = X.solve_pmedian(inst.points, k)
        assert proven
        g["kpp"].append(A.run(d, k, "kmeans++", seed, center="medoid").final_total_distance / opt)
        g["rnd"].append(A.run(d, k, "random", seed, center="medoid").final_total_distance / opt)
        b = E.best_of_restarts(d, k, "medoid", 40, 1).final_total_distance
        g["best"].append(b / opt)
        n_exact += b <= opt * (1 + 1e-9)
        g["kmeans"].append(E.best_of_restarts(d, k, "mean", 40, 1).final_total_distance / opt)
    assert statistics.fmean(g["kpp"]) == PCT(single_kpp, abs=0.0005) and statistics.fmean(g["rnd"]) == PCT(single_rnd, abs=0.0005)
    assert statistics.fmean(g["best"]) == PCT(best, abs=0.0005) and max(g["best"]) == PCT(best_max, abs=0.0005) and n_exact == exact
    assert statistics.fmean(g["kmeans"]) == PCT(kmeans, abs=0.0005) and (min(g["kmeans"]), max(g["kmeans"])) == (PCT(kmeans_min, abs=0.0005), PCT(kmeans_max, abs=0.0005))


def test_moons_are_not_fixed_by_the_medoid():
    """Halbmonde (150 Punkte, k = 2, 20 Netze, bestes von 10 Läufen): Fehlzuordnung Mittelwert 25,4 %, Medoid 23,9 % - beide scheitern."""
    mis = {"mean": [], "medoid": []}
    for seed in list(C.EXPERIMENT_SEEDS)[:20]:
        inst = S.generate_instance(150, 2, 0.1, 0.0, seed, "moons", 0)
        for c in mis:
            r = E.best_of_restarts(inst.as_array(), 2, c, C.N_RESTARTS_COMPARE, C.COMPARE_SEED)
            mis[c].append(E.misassigned_share(r.final_labels, inst.true_labels, 2))
    assert 100 * statistics.fmean(mis["mean"]) == PCT(25.37, abs=0.01) and 100 * statistics.fmean(mis["medoid"]) == PCT(23.90, abs=0.01)


def test_preset_help_numbers_and_defaults_are_consistent():
    assert C.DEFAULT_CENTER == "mean" and C.DEFAULT_OUTLIERS == 0 and set(C.PRESETS["Ausreißer ziehen den Mittelwert"]) >= {"center", "outliers"}
    assert C.PRESETS["Medoid hält gegen Ausreißer"]["outliers"] == C.PRESETS["Ausreißer ziehen den Mittelwert"]["outliers"] == 5
