"""Kennzahlen aus einem Lloyd-Lauf (live, pro Schritt) sowie der Multistart-Vergleich
zwischen Zufalls- und k-Means++-Initialisierung, der die "Wie stark hängt das Ergebnis
vom Zufall ab?"-Sektion der App live berechnet, nicht nur behauptet."""

from dataclasses import dataclass

import numpy as np

from km_algorithm import run
from scipy.optimize import linear_sum_assignment

from km_constants import NEAR_BEST_TOLERANCE


def stats_at_step(result, step):
    """Live-Kennzahlen für die Metrikzeile beim Schritt-Slider / Abspielen."""
    s = result.steps[step]
    is_last_step = step == len(result.steps) - 1
    return {
        "iteration": s.iteration,
        "inertia": s.inertia,
        "objective": s.objective,
        "n_changed": s.n_changed,
        "converged": is_last_step and result.converged,
    }


@dataclass(frozen=True)
class StrategySummary:
    final_inertias: tuple
    seeds: tuple  # gleicher Index wie final_inertias - erlaubt es der App, einzelne Läufe
    # aus genau dieser Stichprobe erneut zu berechnen und darzustellen
    best_inertia: float
    mean_inertia: float
    near_best_fraction: float  # Anteil Läufe höchstens NEAR_BEST_TOLERANCE über dem globalen Besten


@dataclass(frozen=True)
class MultistartComparison:
    strategies: dict  # {"random": StrategySummary, "kmeans++": StrategySummary}
    global_best_inertia: float


def _run_final_inertias(data, k, init_strategy, seeds, center="mean"):
    """Endwerte der Zielgröße des Modus (Inertia beim Mittelwert, Summe der Abstände beim Medoid)."""
    values = []
    for seed in seeds:
        result = run(data, k, init_strategy, int(seed), center=center)
        values.append(result.final_objective)
    return tuple(values)


def multistart_comparison(data, k, n_restarts, base_seed, center="mean"):
    """Führt für beide Init-Strategien je n_restarts unabhängige Läufe auf denselben
    Daten aus und vergleicht die Verteilung der jeweils erreichten (finalen) Inertia. Das
    über beide Strategien gemeinsam beste gefundene Ergebnis dient als praktischer Proxy
    fürs globale Optimum - ein exaktes globales Optimum ist für k-Means NP-schwer zu
    berechnen (siehe Mathe-Abschnitt der App), daher gibt es keinen exakten Referenzlöser."""
    seed_rng = np.random.default_rng(base_seed)
    seeds_random = seed_rng.integers(0, 2**31 - 1, size=n_restarts)
    seeds_kpp = seed_rng.integers(0, 2**31 - 1, size=n_restarts)

    inertias_random = _run_final_inertias(data, k, "random", seeds_random, center)
    inertias_kpp = _run_final_inertias(data, k, "kmeans++", seeds_kpp, center)

    global_best = min(min(inertias_random), min(inertias_kpp))
    threshold = global_best * (1 + NEAR_BEST_TOLERANCE)

    def summarize(inertias, seeds):
        arr = np.array(inertias)
        return StrategySummary(
            final_inertias=inertias,
            seeds=tuple(int(s) for s in seeds),
            best_inertia=float(arr.min()),
            mean_inertia=float(arr.mean()),
            near_best_fraction=float(np.mean(arr <= threshold)),
        )

    return MultistartComparison(
        strategies={
            "random": summarize(inertias_random, seeds_random),
            "kmeans++": summarize(inertias_kpp, seeds_kpp),
        },
        global_best_inertia=global_best,
    )


# --- Mittelwert oder Medoid ----------------------------------------------------------------------------------------------------------

def best_of_restarts(data, k, center, n_restarts, base_seed):
    """Das beste von n_restarts k-Means++-Läufen nach der Zielgröße des Modus (deterministische Seeds ab base_seed)."""
    seeds = np.random.default_rng(base_seed).integers(0, 2**31 - 1, size=n_restarts)
    best = None
    for s in seeds:
        r = run(data, k, "kmeans++", int(s), center=center)
        if best is None or r.final_objective < best.final_objective:
            best = r
    return best


def compare_centers(data, k, n_restarts=10, base_seed=1):
    """Bestes Ergebnis je Modus auf denselben Daten, jeweils in BEIDEN Maßen bewertet: quadrierte Summe (Inertia) und Summe der Abstände."""
    out = {}
    for center in ("mean", "medoid"):
        r = best_of_restarts(data, k, center, n_restarts, base_seed)
        out[center] = r
    return out


def center_shift(centers_a, centers_b):
    """Mittlere Verschiebung der Zentren zwischen zwei Lösungen bei bester Zuordnung (Zuordnungsproblem)."""
    a, b = np.asarray(centers_a, dtype=float), np.asarray(centers_b, dtype=float)
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].mean())


def misassigned_share(labels, true_labels, k):
    """Anteil der Nicht-Ausreißer, die nicht in dem Cluster liegen, das ihrer wahren Gruppe (bei bester Zuordnung der Cluster zu den Gruppen) entspricht."""
    labels = np.asarray(labels)
    true = np.asarray(true_labels)
    keep = true >= 0
    table = np.zeros((k, k), dtype=int)
    for c, g in zip(labels[keep], true[keep]):
        table[c, g] += 1
    rows, cols = linear_sum_assignment(-table)
    return float(1.0 - table[rows, cols].sum() / max(1, keep.sum()))


def robustness(clean, dirty, k, center, n_restarts=10, base_seed=1):
    """Wie stark ändern Ausreißer die Lösung? `clean`, `dirty`: ClusteringInstance ohne und mit Ausreißern (gleiche Grunddaten).
    Verglichen wird das beste Ergebnis (n_restarts k-Means++-Läufe) beider Fälle im selben Modus: Verschiebung der Zentren und Fehlzuordnung der Nicht-Ausreißer."""
    r_clean = best_of_restarts(clean.as_array(), k, center, n_restarts, base_seed)
    r_dirty = best_of_restarts(dirty.as_array(), k, center, n_restarts, base_seed)
    return dict(
        shift=center_shift(r_clean.final_centers, r_dirty.final_centers),
        misassigned=misassigned_share(r_dirty.final_labels, dirty.true_labels, k),
        clean=r_clean, dirty=r_dirty,
    )


def outlier_experiment(n_points, k, spread, imbalance, shape, seeds=None, counts=None, n_restarts=None, base_seed=None):
    """Feste Netze mal Ausreißerzahlen: mittlere und mediane Zentren-Verschiebung, mittlere Fehlzuordnung je Modus und die Zahl der Netze, in denen der Medoid weniger verschoben wird."""
    from km_constants import COMPARE_SEED, EXPERIMENT_SEEDS, N_RESTARTS_COMPARE, OUTLIER_COUNTS
    from km_scenario import generate_instance

    seeds = EXPERIMENT_SEEDS if seeds is None else seeds
    counts = OUTLIER_COUNTS if counts is None else counts
    n_restarts = N_RESTARTS_COMPARE if n_restarts is None else n_restarts
    base_seed = COMPARE_SEED if base_seed is None else base_seed
    out = {}
    for n_out in counts:
        shift = {"mean": [], "medoid": []}
        mis = {"mean": [], "medoid": []}
        for seed in seeds:
            clean = generate_instance(n_points, k, spread, imbalance, seed, shape=shape, n_outliers=0)
            dirty = generate_instance(n_points, k, spread, imbalance, seed, shape=shape, n_outliers=n_out)
            for c in ("mean", "medoid"):
                r = robustness(clean, dirty, k, c, n_restarts, base_seed)
                shift[c].append(r["shift"])
                mis[c].append(r["misassigned"])
        out[n_out] = dict(
            shift_mean={c: float(np.mean(shift[c])) for c in shift}, shift_median={c: float(np.median(shift[c])) for c in shift},
            mis_mean={c: float(np.mean(mis[c])) for c in mis}, mis_max={c: float(np.max(mis[c])) for c in mis},
            medoid_smaller=int(sum(1 for a, b in zip(shift["medoid"], shift["mean"]) if a < b)), count=len(seeds),
        )
    return out
