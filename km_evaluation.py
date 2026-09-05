"""Kennzahlen aus einem Lloyd-Lauf (live, pro Schritt) sowie der Multistart-Vergleich
zwischen Zufalls- und k-Means++-Initialisierung, der die "Wie stark hängt das Ergebnis
vom Zufall ab?"-Sektion der App live berechnet, nicht nur behauptet."""

from dataclasses import dataclass

import numpy as np

from km_algorithm import run
from km_constants import NEAR_BEST_TOLERANCE


def stats_at_step(result, step):
    """Live-Kennzahlen für die Metrikzeile beim Schritt-Slider / Abspielen."""
    s = result.steps[step]
    is_last_step = step == len(result.steps) - 1
    return {
        "iteration": s.iteration,
        "inertia": s.inertia,
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


def _run_final_inertias(data, k, init_strategy, seeds):
    inertias = []
    for seed in seeds:
        result = run(data, k, init_strategy, int(seed))
        inertias.append(result.final_inertia)
    return tuple(inertias)


def multistart_comparison(data, k, n_restarts, base_seed):
    """Führt für beide Init-Strategien je n_restarts unabhängige Läufe auf denselben
    Daten aus und vergleicht die Verteilung der jeweils erreichten (finalen) Inertia. Das
    über beide Strategien gemeinsam beste gefundene Ergebnis dient als praktischer Proxy
    fürs globale Optimum - ein exaktes globales Optimum ist für k-Means NP-schwer zu
    berechnen (siehe Mathe-Abschnitt der App), daher gibt es keinen exakten Referenzlöser."""
    seed_rng = np.random.default_rng(base_seed)
    seeds_random = seed_rng.integers(0, 2**31 - 1, size=n_restarts)
    seeds_kpp = seed_rng.integers(0, 2**31 - 1, size=n_restarts)

    inertias_random = _run_final_inertias(data, k, "random", seeds_random)
    inertias_kpp = _run_final_inertias(data, k, "kmeans++", seeds_kpp)

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
