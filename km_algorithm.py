"""Lloyd's Algorithmus (k-Means) from scratch, mit vollstaendigem Iterations-Protokoll,
damit die App Schritt fuer Schritt (Zuweisung -> Update -> Zuweisung -> ...) durchblaettern
kann - analog zum Knoten-Protokoll von Branch & Bound in branch-bound-demo/bb_solver.py.

Bewusst ohne sklearn zur Laufzeit implementiert, damit jeder Zwischenschritt (nicht nur das
Endergebnis) sichtbar gemacht werden kann. sklearn.cluster.KMeans dient in tests/ nur als
unabhaengiger Kreuzvergleich fuer die eigene Implementierung.
"""

from dataclasses import dataclass

import numpy as np

from km_constants import MAX_ITERATIONS


@dataclass(frozen=True)
class Step:
    iteration: int  # 0 = erste Zuweisung nach der Initialisierung, danach je ein Lloyd-Update
    centers: tuple  # ((x, y), ...), k Eintraege
    labels: tuple  # naechstgelegenes Zentrum je Punkt, Erzeugungsreihenfolge wie die Daten
    inertia: float  # Summe der quadrierten Abstaende zum jeweils zugewiesenen Zentrum (WCSS)
    n_changed: int  # Punkte, die diesen Schritt das Cluster gewechselt haben (Schritt 0: alle)


@dataclass(frozen=True)
class RunResult:
    steps: tuple  # Step-Folge in Ausfuehrungsreihenfolge
    converged: bool  # True, wenn die letzte Zuweisung gegenueber der vorherigen unveraendert blieb
    truncated: bool  # True, wenn max_iter erreicht wurde, ohne dass die Zuweisung stabil wurde

    @property
    def final_step(self):
        return self.steps[-1]

    @property
    def final_inertia(self):
        return self.final_step.inertia

    @property
    def final_centers(self):
        return self.final_step.centers

    @property
    def final_labels(self):
        return self.final_step.labels


def _squared_distances(data, centers):
    diff = data[:, None, :] - centers[None, :, :]
    return (diff ** 2).sum(axis=2)


def _assign(data, centers):
    d2 = _squared_distances(data, centers)
    labels = d2.argmin(axis=1)
    inertia = float(d2[np.arange(len(labels)), labels].sum())
    return labels, inertia


def _update_centers(data, labels, centers, k):
    """Neues Zentrum = Schwerpunkt der zugewiesenen Punkte (minimiert die Summe der
    quadrierten Abweichungen innerhalb des Clusters, siehe Mathe-Abschnitt der App). Ein
    leer gewordenes Cluster behaelt sein bisheriges Zentrum unveraendert - eine gaengige,
    einfache Behandlung dieses Randfalls, die nie einen Punkt komplett verwaist laesst."""
    new_centers = centers.copy()
    for i in range(k):
        mask = labels == i
        if mask.any():
            new_centers[i] = data[mask].mean(axis=0)
    return new_centers


def init_random(data, k, rng):
    """Zufaellige Startpunkte: k verschiedene Datenpunkte gleichverteilt gewaehlt."""
    idx = rng.choice(len(data), size=k, replace=False)
    return data[idx].copy()


def init_kmeans_plusplus(data, k, rng):
    """k-Means++-Seeding (Arthur & Vassilvitskii, 2007): der erste Startpunkt wird
    gleichverteilt gewaehlt, jeder weitere mit Wahrscheinlichkeit proportional zum
    quadrierten Abstand zum naechstgelegenen bereits gewaehlten Zentrum - weit entfernte,
    noch unabgedeckte Punkte werden dadurch bevorzugt zu neuen Startzentren."""
    n = len(data)
    first = rng.integers(n)
    centers = [data[first]]
    for _ in range(1, k):
        current = np.array(centers)
        d2 = _squared_distances(data, current).min(axis=1)
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(n, 1.0 / n)
        idx = rng.choice(n, p=probs)
        centers.append(data[idx])
    return np.array(centers)


INIT_FUNCTIONS = {"random": init_random, "kmeans++": init_kmeans_plusplus}


def run(data, k, init_strategy, seed, max_iter=MAX_ITERATIONS):
    """Fuehrt Lloyd's Algorithmus vollstaendig protokolliert aus: Schritt 0 ist die erste
    Zuweisung nach der Initialisierung, jeder weitere Schritt ist ein vollstaendiger
    Update-dann-Zuweisung-Zyklus. Terminiert, sobald sich die Zuweisung nicht mehr
    aendert (bewiesen endlich, da es nur endlich viele Partitionen gibt und die Inertia
    nie steigt - siehe Mathe-Abschnitt), spaetestens nach max_iter Schritten."""
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=float)
    centers = INIT_FUNCTIONS[init_strategy](data, k, rng)

    labels, inertia = _assign(data, centers)
    steps = [Step(0, tuple(map(tuple, centers)), tuple(int(l) for l in labels), inertia, n_changed=len(data))]

    converged = False
    for iteration in range(1, max_iter + 1):
        new_centers = _update_centers(data, labels, centers, k)
        new_labels, new_inertia = _assign(data, new_centers)
        n_changed = int(np.sum(new_labels != labels))
        steps.append(
            Step(iteration, tuple(map(tuple, new_centers)), tuple(int(l) for l in new_labels), new_inertia, n_changed)
        )
        centers, labels = new_centers, new_labels
        if n_changed == 0:
            converged = True
            break

    truncated = not converged
    return RunResult(steps=tuple(steps), converged=converged, truncated=truncated)
