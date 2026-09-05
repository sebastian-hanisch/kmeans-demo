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
from km_evaluation import multistart_comparison, stats_at_step
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
def _compute_run(n_points, k, spread, imbalance, seed, init_strategy):
    instance = generate_instance(n_points, k, spread, imbalance, seed)
    result = run(instance.as_array(), k, init_strategy, seed)
    return instance, result


@st.cache_data(show_spinner=False)
def _compute_multistart(n_points, k, spread, imbalance, seed):
    instance = generate_instance(n_points, k, spread, imbalance, seed)
    return multistart_comparison(instance.as_array(), k, C.N_RESTARTS_MULTISTART, base_seed=seed)


@st.cache_data(show_spinner=False)
def _compute_example_run(instance, k, init_strategy, example_seed):
    return run(instance.as_array(), k, init_strategy, example_seed)


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
    "Mittlere Schwierigkeit (etwas Überlappung)": "4 Gruppen mit spürbarer Überlappung - der Unterschied zwischen den Start-Strategien wird deutlicher sichtbar.",
    "Schwerer Fall (ungleiche Gruppengrößen)": "5 Gruppen, eine davon groß und diffus, die übrigen klein und dicht - Zufalls-Init scheitert hier reproduzierbar öfter an einem schlechten lokalen Optimum.",
    "Viele Gruppen (Suchraum wächst mit k)": "8 Gruppen - je mehr Depots gesucht werden, desto mehr mögliche Start-Kombinationen gibt es, und desto häufiger trifft reine Zufalls-Init eine schlechte.",
}
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
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

    st.markdown("**Suchverhalten**")
    init_strategy = st.radio(
        "Start-Strategie (für die Animation unten)",
        options=C.INIT_STRATEGIES, key="init_strategy_radio",
        format_func=lambda s: C.INIT_STRATEGY_LABELS[s],
        help="Steuert nur den animierten Einzel-Lauf unten - der '📐'-Vergleich weiter unten "
        "prüft ohnehin immer beide Strategien parallel.",
    )

    st.button(
        "🎲 Neue Punktwolke generieren",
        width="stretch",
        on_click=randomize_seed,
        help="Würfelt einen neuen Zufalls-Seed für die Kundenstandorte.",
    )

sync_query_params(n_points, k, spread, imbalance, seed, init_strategy)

with st.spinner("Führe Lloyd's Algorithmus aus..."):
    instance, result = _compute_run(int(n_points), int(k), spread, imbalance, int(seed), init_strategy)
    comparison = _compute_multistart(int(n_points), int(k), spread, imbalance, int(seed))

max_step = len(result.steps) - 1
run_key = (n_points, k, spread, imbalance, seed, init_strategy)
if "km_step" not in st.session_state or st.session_state.get("km_step_owner") != run_key:
    st.session_state["km_step"] = max_step
    st.session_state["km_step_owner"] = run_key

st.markdown("## 🎯 Lloyd's Algorithmus in Aktion")

step_col, play_col = st.columns([5, 1])
with step_col:
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
    "Inertia (WCSS)", f"{live['inertia']:,.1f}",
    help="Summe der quadrierten Entfernungen jedes Kunden zu seinem aktuell zugewiesenen "
    "Depot - die Größe, die k-Means minimiert.",
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
        example_result = _compute_example_run(instance, int(k), init_strategy, int(example_seed))
        st.plotly_chart(
            build_mini_scatter_figure(instance, example_result), width="stretch", key=f"mini_{example_seed}"
        )
        st.caption(f"Seed {example_seed} · Inertia {example_result.final_inertia:,.1f}")

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
    "Zufällige Startpunkte – Ø finale Inertia", f"{random_summary.mean_inertia:,.1f}",
    delta=f"{gap:,.1f} ggü. k-Means++" if gap != 0 else None, delta_color="inverse",
    help=f"Mittelwert über {C.N_RESTARTS_MULTISTART} unabhängige Läufe mit zufälligen "
    f"Startpunkten. Nur {random_summary.near_best_fraction * 100:.0f}% davon landen nahe "
    f"am besten gefundenen Ergebnis.",
)
mc2.metric(
    "k-Means++ – Ø finale Inertia", f"{kpp_summary.mean_inertia:,.1f}",
    help=f"Mittelwert über {C.N_RESTARTS_MULTISTART} unabhängige Läufe mit k-Means++. "
    f"{kpp_summary.near_best_fraction * 100:.0f}% davon landen nahe am besten gefundenen "
    f"Ergebnis.",
)
mc3.metric(
    "Bestes gefundenes Ergebnis", f"{comparison.global_best_inertia:,.1f}",
    help="Kleinste Inertia über alle Läufe beider Strategien - ein exaktes globales "
    "Optimum ist für k-Means NP-schwer zu berechnen, dies ist der praktische Proxy dafür.",
)

st.plotly_chart(build_multistart_distribution_chart(comparison), width="stretch", key="multistart_distribution")

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

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**k-Means-Zielfunktion:** gegeben $n$ Punkte $x_1, \dots, x_n \in \mathbb{R}^2$ und eine
Partition in $k$ Cluster mit Zentren $c_1, \dots, c_k$, minimiere die **Inertia** (within-
cluster sum of squares, WCSS):

$$
\min_{c_1,\dots,c_k} \sum_{i=1}^n \min_{j \in \{1,\dots,k\}} \lVert x_i - c_j \rVert^2
$$

Bereits für $k=2$ in der Ebene ist das exakte globale Minimum NP-schwer zu berechnen
(Aloise et al., 2009, "NP-hardness of Euclidean sum-of-squares clustering") - deshalb zeigt
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

Implementiert in `km_algorithm.py` (Lloyd's Algorithmus, beide Init-Strategien) und
`km_evaluation.py` (Multistart-Vergleich).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
