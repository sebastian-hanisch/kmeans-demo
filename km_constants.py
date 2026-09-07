"""Defaults, Regler-Grenzen, Sicherheitsgrenzen und Presets für die k-Means-Demo."""

DEFAULT_N_POINTS = 120
DEFAULT_K = 3
DEFAULT_SPREAD = 0.35
DEFAULT_IMBALANCE = 0.0
DEFAULT_SEED = 7
DEFAULT_INIT_STRATEGY = "kmeans++"
DEFAULT_SHAPE = "blobs"

N_POINTS_MIN, N_POINTS_MAX = 30, 400
K_MIN, K_MAX = 2, 8
SPREAD_MIN, SPREAD_MAX = 0.1, 0.9
IMBALANCE_MIN, IMBALANCE_MAX = 0.0, 1.0

SHAPES = ("blobs", "moons")
SHAPE_LABELS = {"blobs": "Gruppen (Blobs)", "moons": "Halbmonde"}

# Hard safety limit so a bad slider combination can never hang the app.
MAX_ITERATIONS = 50

# Live "does randomness matter?" multistart comparison, computed on every rerun.
N_RESTARTS_MULTISTART = 40
NEAR_BEST_TOLERANCE = 0.01  # 1% über der besten gefundenen Inertia gilt noch als "nahe am Besten"

INIT_STRATEGIES = ("random", "kmeans++")
INIT_STRATEGY_LABELS = {
    "random": "Zufällige Startpunkte",
    "kmeans++": "k-Means++",
}

PRESETS = {
    "Einfaches Beispiel (klar getrennte Gruppen)": {
        "n_points": 60, "k": 3, "spread": 0.15, "imbalance": 0.0, "shape": "blobs", "seed": 1,
    },
    "Mittlere Schwierigkeit (etwas Überlappung)": {
        "n_points": 120, "k": 4, "spread": 0.4, "imbalance": 0.0, "shape": "blobs", "seed": 7,
    },
    "Schwerer Fall (ungleiche Gruppengrößen)": {
        "n_points": 150, "k": 5, "spread": 0.15, "imbalance": 0.9, "shape": "blobs", "seed": 1,
    },
    "Viele Gruppen (Suchraum wächst mit k)": {
        "n_points": 240, "k": 8, "spread": 0.4, "imbalance": 0.3, "shape": "blobs", "seed": 11,
    },
    "Nicht-konvexe Formen (k-Means scheitert)": {
        "n_points": 150, "k": 2, "spread": 0.1, "imbalance": 0.0, "shape": "moons", "seed": 2,
    },
}
