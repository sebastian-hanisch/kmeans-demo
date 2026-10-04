"""k-Means für die Standortwahl von Depots - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im
Vergleich) zeigt diese Demo EIN Verfahren - k-Means (Lloyd's Algorithmus) - und lässt
stattdessen die Optimierungslandschaft wachsen: von klar getrennten Gruppen, bei denen
jede Startkonfiguration beim selben Ergebnis landet, bis zu ungleichen, engen Gruppen, bei
denen Zufalls-Init reproduzierbar an schlechten lokalen Optima scheitert. Zweites Stück
der "Konzepte"-Reihe (siehe README für die Einordnung).

Lauffähig mit: streamlit run app.py
"""

import time

import streamlit as st

import km_constants as C
from km_algorithm import run
from km_evaluation import best_of_restarts, compare_centers, multistart_comparison, outlier_experiment, robustness, stats_at_step
from km_exact import solve_pmedian
from km_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from km_scenario import generate_instance
from km_visualization import (
    build_inertia_chart,
    build_mean_vs_medoid_illustration,
    build_mini_scatter_figure,
    build_multistart_distribution_chart,
    build_scatter_figure,
)

st.set_page_config(page_title="k-Means – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _compute_run(n_points, k, spread, imbalance, shape, seed, init_strategy, center="mean", n_outliers=0):
    instance = generate_instance(n_points, k, spread, imbalance, seed, shape=shape, n_outliers=n_outliers)
    result = run(instance.as_array(), k, init_strategy, seed, center=center)
    return instance, result


@st.cache_data(show_spinner=False)
def _compute_multistart(n_points, k, spread, imbalance, shape, seed, center="mean", n_outliers=0):
    instance = generate_instance(n_points, k, spread, imbalance, seed, shape=shape, n_outliers=n_outliers)
    return multistart_comparison(instance.as_array(), k, C.N_RESTARTS_MULTISTART, base_seed=C.COMPARISON_SEED, center=center)


@st.cache_data(show_spinner=False)
def _compute_example_run(instance, k, init_strategy, example_seed, center="mean"):
    return run(instance.as_array(), k, init_strategy, example_seed, center=center)


@st.cache_data(show_spinner=False)
def _compute_compare(instance, k):
    return compare_centers(instance.as_array(), k, C.N_RESTARTS_COMPARE, C.COMPARE_SEED)


@st.cache_data(show_spinner=False)
def _compute_robustness(n_points, k, spread, imbalance, shape, seed, n_outliers):
    clean = generate_instance(n_points, k, spread, imbalance, seed, shape=shape, n_outliers=0)
    dirty = generate_instance(n_points, k, spread, imbalance, seed, shape=shape, n_outliers=n_outliers)
    return {c: robustness(clean, dirty, k, c, C.N_RESTARTS_COMPARE, C.COMPARE_SEED) for c in C.CENTER_TYPES}


@st.cache_data(show_spinner=False)
def _compute_outlier_experiment(n_points, k, spread, imbalance, shape):
    """40 feste Netze mal Ausreißerzahlen: Zentren-Verschiebung und Fehlzuordnung je Modus."""
    return outlier_experiment(n_points, k, spread, imbalance, shape)


@st.cache_data(show_spinner=False)
def _compute_exact(instance, k):
    """Exakter p-Median gegen die Heuristiken auf dem aktuellen Netz."""
    data = instance.as_array()
    opt, centers, proven = solve_pmedian(instance.points, k)
    return dict(
        opt=opt, proven=proven,
        best_medoid=best_of_restarts(data, k, "medoid", C.N_RESTARTS_MULTISTART, C.COMPARE_SEED).final_total_distance,
        best_mean=best_of_restarts(data, k, "mean", C.N_RESTARTS_MULTISTART, C.COMPARE_SEED).final_total_distance,
        single_medoid=run(data, k, "kmeans++", C.COMPARE_SEED, center="medoid").final_total_distance,
        single_random=run(data, k, "random", C.COMPARE_SEED, center="medoid").final_total_distance,
    )


st.title("📍 k-Means für die Standortwahl von Depots")
st.markdown(
    """
Kundenstandorte sollen auf **k Depots** aufgeteilt werden, jedes Depot bedient die ihm am
nächsten liegenden Kunden - gesucht ist die Aufteilung, die die Summe der quadrierten
Entfernungen zwischen Kunde und zuständigem Depot minimiert. Das ist exakt das, was
**k-Means** berechnet: eine Optimierungsheuristik, die abwechselnd Kunden dem nächsten
Depot zuordnet und jedes Depot in den Schwerpunkt seiner zugeordneten Kunden verschiebt
(**Lloyd's Algorithmus**). Genau **wie** das funktioniert, erklärt der aufgeklappte
Abschnitt direkt darunter - bevor weiter unten die Suche live dazu läuft und die Frage
"📐 Wie stark hängt das Ergebnis vom Zufall der Startpunkte ab?" live beantwortet wird.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren "
    "vergleichen, zeigt diese Demo - Teil der wachsenden \"Konzepte\"-Reihe - **ein** Verfahren "
    "an einem wachsenden Beispiel: k-Means selbst ist eine Optimierungsheuristik (kein exakter "
    "Löser) und kann, wie jede Local-Search-Methode im Portfolio, in einem lokalen Optimum "
    "hängen bleiben."
)

with st.expander("So funktioniert k-Means", expanded=True):
    st.markdown(
        """
Lloyd's Algorithmus wiederholt zwei einfache Schritte, bis sich nichts mehr ändert:

1. **Zuweisung**: jeder Kunde wird dem näher gelegenen der aktuellen Depot-Standorte
   zugeordnet.
2. **Update**: jedes Depot wandert an den Schwerpunkt (Mittelwert) der ihm gerade
   zugeordneten Kunden - das ist nachweislich die Position, die die Summe der quadrierten
   Entfernungen innerhalb dieser Gruppe minimiert (siehe "📐 Mathematische Formulierung").

Jeder Durchlauf verbessert die Zielfunktion (oder lässt sie gleich) - sie kann nie wieder
schlechter werden. Das Verfahren **konvergiert** deshalb garantiert, aber Konvergenz ist
**keine Garantie fürs globale Optimum**: je nachdem, wo die Depots zu Beginn stehen, kann
das Verfahren in einer schlechteren, aber stabilen Aufteilung "hängen bleiben". Zwei
Start-Strategien stehen zur Wahl:

- **Zufällige Startpunkte**: k Kunden werden gleichverteilt als erste Depot-Standorte
  gewählt - einfach, aber es kann passieren, dass zwei Startpunkte in dieselbe Kundengruppe
  fallen und eine andere Gruppe zunächst ganz ohne "eigenes" Depot bleibt.
- **k-Means++**: der erste Startpunkt wird zufällig gewählt, jeder weitere bevorzugt einen
  Kunden, der **weit von allen bisherigen Startpunkten entfernt** liegt - dadurch landet in
  der Praxis viel öfter genau ein Startpunkt pro tatsächlicher Gruppe.

Die Punktwolke weiter unten zeigt das live: Punkte sind nach aktueller Zuordnung eingefärbt,
Sterne markieren die aktuellen Depot-Standorte.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
PRESET_HELP = {
    "Einfaches Beispiel (klar getrennte Gruppen)": "3 klar getrennte, gleich große Gruppen - k-Means++ trifft praktisch immer die beste Aufteilung, reine Zufalls-Init kann aber auch hier schon danebengreifen.",
    "Mittlere Schwierigkeit (etwas Überlappung)": "4 Gruppen mit spürbarer Überlappung - hier liegen die beiden Start-Strategien dicht beieinander; erst ungleiche Gruppengrößen machen den Unterschied deutlich.",
    "Schwerer Fall (ungleiche Gruppengrößen)": "5 Gruppen, eine davon groß und diffus, die übrigen klein und dicht - Zufalls-Init scheitert hier reproduzierbar öfter an einem schlechten lokalen Optimum.",
    "Viele Gruppen (Suchraum wächst mit k)": "8 Gruppen - je mehr Depots gesucht werden, desto mehr mögliche Start-Kombinationen gibt es, und desto häufiger trifft reine Zufalls-Init eine schlechte.",
    "Ausreißer ziehen den Mittelwert": "120 Punkte, 3 Gruppen, 5 Ausreißer: der Mittelwert gibt ein Zentrum an die Ausreißer ab und verschmilzt zwei echte Gruppen - im besten von 10 Läufen wandern die Zentren im Mittel um 4,87 gegenüber der Lösung ohne Ausreißer, ein Drittel der echten Punkte ist falsch zugeordnet.",
    "Medoid hält gegen Ausreißer": "Dieselben Daten mit dem Medoid als Zentrum: die Zentren bleiben in den echten Gruppen (Verschiebung 0,00, keine falsche Zuordnung). Summe der Abstände 176,3 gegen 273,6 beim Mittelwert; in der Inertia dafür 1 004,5 gegen 873,7 - jedes Verfahren gewinnt in seinem eigenen Maß.",
    "Nicht-konvexe Formen (k-Means scheitert)": "Zwei ineinander verschlungene Halbmonde - k-Means kann sie grundsätzlich nicht sauber trennen, unabhängig von der Start-Strategie, weil beide Gruppen nicht konvex sind.",
}
preset_names = list(C.PRESETS.keys())
for row_start in range(0, len(preset_names), 4):
    preset_cols = st.columns(4)
    for col, name in zip(preset_cols, preset_names[row_start:row_start + 4]):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_points = st.slider("Anzahl Kunden", *bounds("n_points_slider"), key="n_points_slider")
    k = st.slider("Anzahl Depots (k)", *bounds("k_slider"), key="k_slider")
    spread = st.slider(
        "Streuung / Überlappung", *bounds("spread_slider"), key="spread_slider", step=0.05,
        help="Klein = Gruppen klar getrennt. Groß = Gruppen überlappen sich spürbar.",
    )
    imbalance = st.slider(
        "Größen-Ungleichgewicht", *bounds("imbalance_slider"), key="imbalance_slider", step=0.05,
        help="0 = alle Gruppen gleich groß und gleich dicht. 1 = eine Gruppe wird groß und "
        "diffus, die übrigen klein und dicht - der klassische Fall, in dem Zufalls-Init "
        "besonders oft an einem schlechten lokalen Optimum hängen bleibt.",
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)

    st.markdown("**Punktwolken-Form**")
    shape = st.radio(
        "Form", options=C.SHAPES, key="shape_radio", format_func=lambda s: C.SHAPE_LABELS[s],
        help="„Gruppen“: runde, konvexe Cluster - genau die Annahme, auf der k-Means beruht. "
        "„Halbmonde“: nicht-konvexe Bögen, die k-Means grundsätzlich nicht sauber trennen "
        "kann (siehe dbscan-demo/spectral-demo für Verfahren, die das beheben).",
    )

    st.markdown("**Suchverhalten**")
    init_strategy = st.radio(
        "Start-Strategie (für die Animation unten)",
        options=C.INIT_STRATEGIES, key="init_strategy_radio",
        format_func=lambda s: C.INIT_STRATEGY_LABELS[s],
        help="Steuert nur den animierten Einzel-Lauf unten - der '📐'-Vergleich weiter unten "
        "prüft ohnehin immer beide Strategien parallel.",
    )

    st.markdown("**Zentrum jedes Clusters**")
    center = st.radio(
        "Zentrum", options=C.CENTER_TYPES, key="center_radio", format_func=lambda c: C.CENTER_LABELS[c],
        help="Mittelwert: der Schwerpunkt der zugewiesenen Punkte (k-Means, minimiert die Summe quadrierter Abstände). Medoid: der zentralste echte Datenpunkt "
        "des Clusters (k-Medoids, minimiert die Summe der Abstände) - robuster gegen Ausreißer, aber pro Update teurer.",
    )
    n_outliers = st.slider(
        "Ausreißer", *bounds("outliers_slider"), key="outliers_slider",
        help="Fügt weit entfernte einzelne Punkte hinzu (außerhalb der Gruppen). Sie zeigen, wie stark der Mittelwert gegenüber dem Medoid verzerrt wird.",
    )

    st.button(
        "🎲 Neue Punktwolke generieren",
        width="stretch",
        on_click=randomize_seed,
        help="Würfelt einen neuen Zufalls-Seed für die Kundenstandorte.",
    )

sync_query_params(n_points, k, spread, imbalance, seed, shape, init_strategy, center, n_outliers)

with st.spinner("Führe Lloyd's Algorithmus aus..."):
    instance, result = _compute_run(int(n_points), int(k), spread, imbalance, shape, int(seed), init_strategy, center, int(n_outliers))
    comparison = _compute_multistart(int(n_points), int(k), spread, imbalance, shape, int(seed), center, int(n_outliers))
OBJ = C.OBJECTIVE_LABELS[center]

max_step = len(result.steps) - 1
run_key = (n_points, k, spread, imbalance, shape, seed, init_strategy, center, n_outliers)
if "km_step" not in st.session_state or st.session_state.get("km_step_owner") != run_key:
    st.session_state["km_step"] = max_step
    st.session_state["km_step_owner"] = run_key

st.markdown("## 🎯 Lloyd's Algorithmus in Aktion")

step_col, play_col = st.columns([5, 1])
with step_col:
    if max_step == 0:
        step = 0
        st.caption("Bereits nach der ersten Zuweisung konvergiert - kein Regler nötig.")
    else:
        step = st.slider(
            "Schritt (Iteration)", 0, max_step, key="km_step",
            help="Schritt 0 = erste Zuweisung nach der Initialisierung, danach je ein "
            "vollständiger Update-dann-Zuweisung-Zyklus.",
        )
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")

scatter_slot = st.empty()
inertia_slot = st.empty()


def _render(current_step):
    scatter_slot.plotly_chart(
        build_scatter_figure(instance, result, current_step), width="stretch", key=f"scatter_{current_step}"
    )
    inertia_slot.plotly_chart(
        build_inertia_chart(result, current_step), width="stretch", key=f"inertia_{current_step}"
    )


if auto_play:
    n_frames = min(max_step + 1, 30)
    frame_skip = max(1, (max_step + 1) // n_frames)
    for s in list(range(0, max_step, frame_skip)) + [max_step]:
        _render(s)
        time.sleep(0.35)
    step = max_step
else:
    _render(step)

live = stats_at_step(result, step)
lm1, lm2, lm3, lm4 = st.columns(4)
lm1.metric("Iteration", live["iteration"])
lm2.metric(
    OBJ, f"{live['objective']:,.1f}",
    help=(
        "Summe der quadrierten Entfernungen jedes Kunden zu seinem aktuell zugewiesenen Depot - die Größe, die k-Means minimiert."
        if center == "mean" else
        "Summe der Entfernungen (nicht quadriert) jedes Kunden zu seinem aktuell zugewiesenen Depot - die Größe, die k-Medoids minimiert."
    ),
)
lm3.metric(
    "Kunden, die dieses Depot gewechselt haben", live["n_changed"],
    help="0 bedeutet: die Zuordnung ist stabil, das Verfahren hat konvergiert.",
)
lm4.metric("Konvergiert?", "Ja" if live["converged"] else "Nein")

if result.truncated:
    st.error(
        f"⛔ Nach {C.MAX_ITERATIONS} Iterationen noch nicht konvergiert - ungewöhnlich für "
        f"k-Means, das gezeigte Ergebnis ist der Zwischenstand, nicht der stabile Endzustand."
    )

st.markdown("**Und mit anderen Zufalls-Seeds?**")
st.caption(
    f"Gleiche Kundenstandorte, gleiche Start-Strategie ({C.INIT_STRATEGY_LABELS[init_strategy]}) "
    "wie oben - nur der Zufalls-Seed der Initialisierung unterscheidet sich. Diese Läufe stammen "
    "aus genau der Stichprobe, die die Verteilung weiter unten zusammenfasst."
)
example_seeds = comparison.strategies[init_strategy].seeds[:4]
example_cols = st.columns(len(example_seeds))
for col, example_seed in zip(example_cols, example_seeds):
    with col:
        example_result = _compute_example_run(instance, int(k), init_strategy, int(example_seed), center)
        st.plotly_chart(
            build_mini_scatter_figure(instance, example_result), width="stretch", key=f"mini_{example_seed}"
        )
        st.caption(f"Seed {example_seed} · {OBJ} {example_result.final_objective:,.1f}")

st.markdown("---")

st.subheader("📐 Wie stark hängt das Ergebnis vom Zufall der Startpunkte ab?")
st.markdown(
    f"""
Beide Start-Strategien konvergieren garantiert (siehe oben) - der Unterschied liegt allein
darin, **wo** sie landen. Live für Ihr aktuelles Szenario mit je
**{C.N_RESTARTS_MULTISTART} unabhängigen Läufen** pro Strategie geprüft, nicht nur
behauptet:
"""
)

random_summary = comparison.strategies["random"]
kpp_summary = comparison.strategies["kmeans++"]
gap = random_summary.mean_inertia - kpp_summary.mean_inertia

mc1, mc2, mc3 = st.columns(3)
mc1.metric(
    f"Zufällige Startpunkte – Ø {OBJ}", f"{random_summary.mean_inertia:,.1f}",
    delta=f"{gap:,.1f} ggü. k-Means++" if gap != 0 else None, delta_color="inverse",
    help=f"Mittelwert über {C.N_RESTARTS_MULTISTART} unabhängige Läufe mit zufälligen "
    f"Startpunkten. Nur {random_summary.near_best_fraction * 100:.0f}% davon landen nahe "
    f"am besten gefundenen Ergebnis.",
)
mc2.metric(
    f"k-Means++ – Ø {OBJ}", f"{kpp_summary.mean_inertia:,.1f}",
    help=f"Mittelwert über {C.N_RESTARTS_MULTISTART} unabhängige Läufe mit k-Means++. "
    f"{kpp_summary.near_best_fraction * 100:.0f}% davon landen nahe am besten gefundenen "
    f"Ergebnis.",
)
mc3.metric(
    "Bestes gefundenes Ergebnis", f"{comparison.global_best_inertia:,.1f}",
    help="Kleinste Inertia über alle Läufe beider Strategien - ein exaktes globales "
    "Optimum ist für k-Means NP-schwer zu berechnen, dies ist der praktische Proxy dafür.",
)

st.plotly_chart(build_multistart_distribution_chart(comparison, center), width="stretch", key="multistart_distribution")

if gap > comparison.global_best_inertia * 0.05:
    st.success(
        f"✅ Bei diesem Szenario liegt Zufalls-Init im Mittel **{gap:,.1f}** über "
        f"k-Means++ - {random_summary.near_best_fraction * 100:.0f}% der Zufalls-Läufe "
        f"gegenüber {kpp_summary.near_best_fraction * 100:.0f}% der k-Means++-Läufe "
        f"landen nahe am besten gefundenen Ergebnis. Der Unterschied liegt komplett in der "
        f"Startstrategie, nicht im Algorithmus selbst."
    )
else:
    st.info(
        "Bei diesem (einfachen) Szenario ist der Unterschied noch klein - ein größeres "
        "Größen-Ungleichgewicht oder mehr Gruppen (Regler links) macht ihn deutlicher."
    )

st.markdown("---")

st.subheader("📐 Mittelwert oder Medoid? Was ändern Ausreißer?")
st.markdown(
    """
k-Means setzt das Zentrum auf den **Mittelwert** und minimiert die Summe **quadrierter** Abstände, k-Medoids auf den **Medoid** (den zentralsten echten Datenpunkt) und minimiert die Summe der Abstände.
Quadrierte Abstände gewichten weit entfernte Punkte stark: ein Ausreißer zieht den Mittelwert, ein Medoid bleibt in der dichten Gruppe. Hier auf denselben Daten wie oben (**"""
    + str(int(n_points)) + """ Punkte plus """ + str(int(n_outliers)) + """ Ausreißer**), jeweils das beste von """ + str(C.N_RESTARTS_COMPARE) + """ k-Means++-Läufen, bewertet in **beiden** Maßen:
"""
)
cmp_results = _compute_compare(instance, int(k))
cmp_cols = st.columns(2)
for col, c_type in zip(cmp_cols, C.CENTER_TYPES):
    r = cmp_results[c_type]
    with col:
        st.markdown(f"**{C.CENTER_LABELS[c_type]}**")
        st.plotly_chart(build_mini_scatter_figure(instance, r), width="stretch", key=f"cmp_{c_type}")
        st.metric("Inertia (Summe quadrierter Abstände)", f"{r.final_step.inertia:,.1f}")
        st.metric("Summe der Abstände", f"{r.final_step.total_distance:,.1f}")
st.caption(
    "Jedes Verfahren ist in seinem eigenen Maß besser: der Mittelwert bei der quadrierten Summe (Inertia), der Medoid bei der Summe der Abstände - wobei die Zentren des Mittelwerts frei liegen dürfen, die des Medoids Datenpunkte sein müssen."
)

if int(n_outliers) > 0:
    rob = _compute_robustness(int(n_points), int(k), spread, imbalance, shape, int(seed), int(n_outliers))
    st.markdown(f"**Wirkung der {int(n_outliers)} Ausreißer** (gegen dieselbe Lösung ohne Ausreißer, bestes von {C.N_RESTARTS_COMPARE} Läufen):")
    st.table({
        "Zentrum": [C.CENTER_LABELS[c_type] for c_type in C.CENTER_TYPES],
        "mittlere Verschiebung der Zentren": [f"{rob[c_type]['shift']:.2f}" for c_type in C.CENTER_TYPES],
        "falsch zugeordnete echte Punkte": [f"{100 * rob[c_type]['misassigned']:.1f} %" for c_type in C.CENTER_TYPES],
    })
else:
    st.info("Stellen Sie links **Ausreißer** auf einen Wert über 0: dann zeigt die Tabelle, wie weit die Zentren durch sie wandern und wie viele echte Punkte falsch zugeordnet werden.")

st.markdown("**Gilt das auch im Mittel?** 40 feste Netze (gleiche Einstellungen wie links, ohne den Seed) mit 1, 3, 5 und 10 Ausreißern:")
if st.button("40 Netze durchrechnen (dauert einige Sekunden)", key="outlier_exp_start"):
    st.session_state["outlier_exp_on"] = True
if st.session_state.get("outlier_exp_on"):
    with st.spinner("Rechne 40 Netze × 4 Ausreißerzahlen × 2 Verfahren..."):
        exp = _compute_outlier_experiment(int(n_points), int(k), spread, imbalance, shape)
    st.table({
        "Ausreißer": [str(n) for n in exp],
        "Verschiebung Mittelwert (Ø / Median)": [f"{exp[n]['shift_mean']['mean']:.2f} / {exp[n]['shift_median']['mean']:.2f}" for n in exp],
        "Verschiebung Medoid (Ø / Median)": [f"{exp[n]['shift_mean']['medoid']:.2f} / {exp[n]['shift_median']['medoid']:.2f}" for n in exp],
        "Fehlzuordnung Mittelwert / Medoid": [f"{100 * exp[n]['mis_mean']['mean']:.1f} % / {100 * exp[n]['mis_mean']['medoid']:.1f} %" for n in exp],
        "Medoid weniger verschoben in": [f"{exp[n]['medoid_smaller']} von {exp[n]['count']} Netzen" for n in exp],
    })
    st.caption("Verschoben wird nur, was ein Ausreißer anrichtet: der Mittelwert wandert mit ihm oder gibt ein Zentrum an ihn ab (dann verschmelzen zwei echte Gruppen), der Medoid bleibt meist bei den echten Punkten.")

st.markdown("**Wie gut ist die Medoid-Heuristik? Der exakte p-Median als Referenz.** k-Medoids ist das **p-Median-Problem** der Standortplanung (Zentren sind Punkte, die Summe der Abstände ist minimal): dort per MILP exakt lösbar, hier bis " + str(C.EXACT_MAX_POINTS) + " Punkte auf Abruf.")
exact_blocked = len(instance.points) > C.EXACT_MAX_POINTS
if st.button(
    "Exakt nachrechnen (p-Median per MILP)", key="exact_start", disabled=exact_blocked,
    help=f"Nur bis {C.EXACT_MAX_POINTS} Punkte (einschließlich Ausreißer): darüber wird das MILP zu groß für die App." if exact_blocked else "Löst das p-Median-Problem auf den aktuellen Punkten exakt (HiGHS).",
):
    st.session_state["exact_on"] = True
if exact_blocked:
    st.caption(f"Bei mehr als {C.EXACT_MAX_POINTS} Punkten (aktuell {len(instance.points)}) ist die exakte Rechnung nicht verfügbar - Regler „Anzahl Kunden“ oder „Ausreißer“ verkleinern.")
elif st.session_state.get("exact_on"):
    with st.spinner("Löse den p-Median exakt..."):
        ex = _compute_exact(instance, int(k))
    st.table({
        "Verfahren": ["Exakt (p-Median, Zentren = Punkte)", f"k-Medoids, bestes von {C.N_RESTARTS_MULTISTART} Läufen", "k-Medoids, ein Lauf (k-Means++-Start)", "k-Medoids, ein Lauf (Zufalls-Start)", f"k-Means, bestes von {C.N_RESTARTS_MULTISTART} (Mittelwert-Zentren)"],
        "Summe der Abstände": [f"{ex['opt']:,.1f}", f"{ex['best_medoid']:,.1f}", f"{ex['single_medoid']:,.1f}", f"{ex['single_random']:,.1f}", f"{ex['best_mean']:,.1f}"],
        "gegenüber dem Optimum": ["1,000"] + [f"{v / ex['opt']:.3f}".replace(".", ",") for v in (ex["best_medoid"], ex["single_medoid"], ex["single_random"], ex["best_mean"])],
    })
    st.caption("Die Medoid-Heuristik (Voronoi-Iteration) ist ein lokales Verfahren; mit mehreren Starts nähert sie sich dem Optimum. Die k-Means-Zentren dürfen zwischen den Punkten liegen und können deshalb sogar unter dem p-Median-Optimum liegen (Werte unter 1,000).")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**k-Means-Zielfunktion:** gegeben $n$ Punkte $x_1, \dots, x_n \in \mathbb{R}^2$ und eine
Partition in $k$ Cluster mit Zentren $c_1, \dots, c_k$, minimiere die **Inertia** (within-
cluster sum of squares, WCSS):

$$
\min_{c_1,\dots,c_k} \sum_{i=1}^n \min_{j \in \{1,\dots,k\}} \lVert x_i - c_j \rVert^2
$$

Bereits für $k=2$ ist das exakte globale Minimum NP-schwer zu berechnen, wenn die Dimension nicht
beschränkt ist (Aloise et al., 2009, "NP-hardness of Euclidean sum-of-squares clustering"); auch in der
Ebene bleibt es für beliebiges $k$ NP-schwer (Mahajan, Nimbhorkar, Varadarajan, "The planar k-means
problem is NP-hard") - deshalb zeigt
diese Demo bewusst keinen exakten Löser, sondern die Heuristik, mit der k-Means in der
Praxis tatsächlich verwendet wird.

**Lloyd's Algorithmus** wiederholt zwei Schritte bis zur Konvergenz:

1. **Zuweisung** (Voronoi-Partition bei festen Zentren): $\text{label}(x_i) = \arg\min_j
   \lVert x_i - c_j \rVert^2$.
2. **Update**: $c_j \leftarrow \frac{1}{|S_j|} \sum_{x_i \in S_j} x_i$, wobei $S_j$ die
   Punkte mit $\text{label}(x_i) = j$ sind - der Mittelwert minimiert nachweislich
   $\sum_{x_i \in S_j} \lVert x_i - c_j \rVert^2$ innerhalb dieser Gruppe (Ableitung nach
   $c_j$ gleich Null setzen ergibt genau den Mittelwert).

**Mittelwert vs. Medoid:** ersetzt man in Schritt 2 den Mittelwert durch den
nächstgelegenen tatsächlichen Datenpunkt, erhält man **k-Medoids** - mit beliebigen
Distanzmaßen nutzbar, aber teurer pro Iteration ($O(|S_j|^2)$ statt $O(|S_j|)$ für den
Mittelwert, da die Medoid-Suche alle Punkte im Cluster gegeneinander prüfen muss). Der
eigentliche Grund, das überhaupt in Betracht zu ziehen: der Mittelwert geht quadratisch in
die Zielfunktion ein, ein einzelner extremer Ausreißer kann ihn dadurch beliebig weit von
der eigentlichen Punktgruppe wegziehen. Der Medoid wird über die Summe der (nicht
quadrierten) Abstände zu allen anderen Punkten der Gruppe bestimmt - ein Ausreißer kann
diese Summe nicht annähernd so leicht dominieren.

Ob das wirklich so kommt, lässt sich ausprobieren: in der Seitenleiste das **Zentrum** auf Medoid umstellen und **Ausreißer** hinzufügen; weiter unten vergleicht der Abschnitt „Mittelwert oder Medoid?“ beide Verfahren auf denselben Daten.

Ein Beispiel macht das greifbar: fünf eng beieinanderliegende Punkte plus ein einzelner,
weit entfernter Ausreißer. Der Mittelwert aller sechs Punkte liegt bei (3.75, 3.75) - mitten
im Leeren, weit außerhalb der eigentlichen Gruppe, weil der Ausreißer ihn dorthin zieht. Der
Medoid bleibt bei (0.5, 0.5), einem echten Punkt mitten in der dichten Gruppe, weil er dort
die kleinste Summe an Abständen zu allen anderen erreicht:
        """
    )
    st.plotly_chart(build_mean_vs_medoid_illustration(), width="stretch", key="mean_vs_medoid")
    st.markdown(
        r"""
**Konvergenz:** jeder der beiden Schritte kann die Zielfunktion nur verkleinern oder
gleich lassen, nie vergrößern. Da es für $n$ Punkte nur endlich viele Partitionen in $k$
Gruppen gibt, kann sich keine Partition wiederholen, ohne dass der Algorithmus terminiert -
Lloyd's Algorithmus konvergiert deshalb garantiert in endlich vielen Schritten
(siehe `tests/test_algorithm.py`, das die Monotonie über viele
Zufallsinstanzen prüft). Konvergenz ist dabei ausdrücklich **nur** ein Nachweis, dass ein
**lokales** Optimum erreicht ist - welches, hängt von der Startkonfiguration ab (siehe
`test_bad_init_can_converge_to_a_worse_local_optimum`, das genau diesen Fall von Hand
nachrechnet).

**k-Means++-Seeding** (Arthur & Vassilvitskii, 2007): der erste Startpunkt wird
gleichverteilt gewählt, jeder weitere Punkt $x$ mit Wahrscheinlichkeit proportional zu
$D(x)^2$, wobei $D(x)$ der Abstand zum nächstgelegenen bereits gewählten Zentrum ist:

$$
P(x) = \frac{D(x)^2}{\sum_{x'} D(x')^2}
$$

Dieses Seeding-Schema garantiert (in Erwartung) eine $O(\log k)$-Approximation des
globalen Optimums - eine beweisbare Garantie, die reine Zufalls-Init nicht hat.

**k-Medoids** (Zentrum = Medoid): minimiere die Summe der (nicht quadrierten) Abstände

$$
\min_{S \subseteq X,\ |S| = k} \sum_{i=1}^n \min_{c \in S} \lVert x_i - c \rVert
$$

über Zentren, die Datenpunkte sind (das **p-Median-Problem**). Die hier gerechnete Heuristik ist die Voronoi-Iteration: Zuweisung zum nächsten Zentrum, dann je Cluster der Punkt
mit der kleinsten Abstandssumme zu den übrigen Punkten des Clusters als neues Zentrum ($O(|S_j|^2)$ je Update). Beide Schritte senken die Summe der Abstände nie, das Verfahren
konvergiert also, aber wie Lloyd nur zu einem lokalen Optimum; der exakte p-Median ist per MILP lösbar (bis 80 Punkte auf Abruf).

Implementiert in `km_algorithm.py` (Lloyd's Algorithmus mit Mittelwert oder Medoid, beide Init-Strategien), `km_evaluation.py` (Multistart-Vergleich, Ausreißer-Experiment) und
`km_exact.py` (exakter p-Median).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Clustering erklärt: k-Means bis HDBSCAN](https://sebastianhanisch.net/konzepte-clustering.html)."
)
