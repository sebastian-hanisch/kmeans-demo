# k-Means für die Standortwahl von Depots – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-kmeans-demo.streamlit.app/)**

Zweites Stück der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research
und Machine Learning" (nach [branch-bound-demo](../branch-bound-demo)): anders als die
Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese
Demo **ein** Verfahren – k-Means (Lloyd's Algorithmus) – und lässt stattdessen die
**Optimierungslandschaft** wachsen. Vehikel-Problem: Kundenstandorte auf k Depots
aufteilen, jedes Depot bedient die ihm am nächsten liegenden Kunden – die Summe der
quadrierten Entfernungen zu minimieren ist exakt die k-Means-Zielfunktion.

Die Konzepte-Reihe ist kein linearer Pfad, sondern mehrere unabhängige Linien: diese Demo
ist der Startpunkt einer eigenen **Clustering-Linie**, unabhängig von
[branch-bound-demo](../branch-bound-demo)s Exakte-Suche-Linie - und verzweigt sich hier in
zwei unabhängige Äste:

```
kmeans-demo → dbscan-demo ──┐
                             ├──> hdbscan-demo
              agglomerative-demo ──────────┘
kmeans-demo → gmm-demo → dpmm-demo
kmeans-demo → spectral-demo → leiden-demo
kmeans-demo ─┐
             ├──> divisive-demo
agglomerative-demo ─┘
```

Ein Ast behebt k-Means' **dichte-/verbindungsbasierte** Schwächen: [dbscan-demo](../dbscan-demo)
(k muss nicht vorab feststehen, Cluster müssen nicht konvex/kugelförmig sein) und, davon
unabhängig, [agglomerative-demo](../agglomerative-demo) (Chaining bei Single-Linkage) -
beide laufen auf [hdbscan-demo](../hdbscan-demo) zu, das live nachweist, dass HDBSCAN beide
Probleme löst. Ein zweiter, unabhängiger Ast behebt eine ANDERE Schwäche - die implizite
Annahme kugelförmiger, gleich gestreuter Cluster und harter Zuweisung:
[gmm-demo](../gmm-demo) (Gaussian Mixture Models, EM-Algorithmus), bei dem k-Means selbst
der Grenzfall ist, fortgesetzt von [dpmm-demo](../dpmm-demo) (Dirichlet-Process-Mixture),
das auch dort noch die feste Clusterzahl k über Bayesianische Nichtparametrik auflöst. Ein
dritter, unabhängiger Ast behebt NUR die Nicht-Konvexitäts-Annahme über einen völlig
anderen Werkzeugkasten: [spectral-demo](../spectral-demo) (Spectral Clustering, Graph-
Laplace-Matrix + Eigenzerlegung) - ein ehrlicher Kontrast zu DBSCAN, da k dabei weiterhin
vorab feststehen muss. [leiden-demo](../leiden-demo) setzt diesen Ast fort und behebt
auch dort noch die feste Clusterzahl k - über Modularitätsoptimierung (Leiden-
Algorithmus, dem De-facto-Standard der Netzwerk-Community-Detection) statt über
Dichte oder Bayesianische Nichtparametrik. [divisive-demo](../divisive-demo) (Bisecting
k-Means) behebt gar keine Schwäche - es ist ein bewusster Kontrast zu
agglomerative-demo: dieselbe Art Hierarchie, top-down statt bottom-up gebaut, mit
eigenen Kompromissen.

## Warum diese Demo anders aufgebaut ist

Die Fall-Demos im Portfolio beantworten "welches Verfahren löst diesen einen Fall am
besten?". Diese Demo (wie die übrigen Stücke der "Konzepte"-Reihe) beantwortet stattdessen
"wie verhält sich EIN Verfahren, wenn das Beispiel tückischer wird?" – bei klar getrennten,
gleich großen Kundengruppen landet praktisch jede Startkonfiguration beim selben Ergebnis,
bei ungleichen, engen Gruppen scheitert Zufalls-Initialisierung reproduzierbar an
schlechten lokalen Optima. Genau dieser Kontrast ist der Punkt, nicht ein
Methodenvergleich.

Anders als bei branch-bound-demo (Baumgröße als Schwierigkeitsachse) wächst hier **wie
tückisch die Optimierungslandschaft ist**, gesteuert über drei unabhängige Regler:

- **Anzahl Depots (k)** – mehr gesuchte Cluster bedeuten mehr mögliche
  Start-Kombinationen und damit mehr Gelegenheiten für ein schlechtes lokales Optimum.
- **Streuung / Überlappung** – klein: Kundengruppen klar getrennt. Groß: Gruppen
  überlappen sich spürbar.
- **Größen-Ungleichgewicht** – 0: alle Gruppen gleich groß und gleich dicht. 1: eine
  Gruppe wird groß und diffus, die übrigen klein und dicht – der klassische Fall, in dem
  Zufalls-Init eine kleine Gruppe komplett übersieht.

## Lloyd's Algorithmus und die zwei Start-Strategien

Jede Iteration besteht aus einer Zuweisung (jeder Kunde zum nächstgelegenen Depot) gefolgt
von einem Update (jedes Depot wandert an den Schwerpunkt seiner Kunden). Das Verfahren
konvergiert **garantiert** – die Zielfunktion (Inertia, Summe der quadrierten Entfernungen)
fällt bei jedem Schritt monoton, und da es nur endlich viele mögliche Partitionen gibt,
kann sich keine Partition wiederholen, ohne dass der Algorithmus terminiert. Konvergenz
ist aber ausdrücklich **keine Garantie fürs globale Optimum** – wohin das Verfahren
konvergiert, hängt von den Startpunkten ab:

- **Zufällige Startpunkte**: k Kunden gleichverteilt gewählt – kann zwei Startpunkte in
  dieselbe Gruppe legen und eine andere Gruppe zunächst unbesetzt lassen.
- **k-Means++**: der erste Startpunkt zufällig, jeder weitere mit Wahrscheinlichkeit
  proportional zum quadrierten Abstand zum nächstgelegenen bereits gewählten Zentrum
  (Arthur & Vassilvitskii, 2007) – landet in der Praxis viel öfter mit genau einem
  Startpunkt pro tatsächlicher Gruppe, mit einer beweisbaren O(log k)-Approximationsgarantie.

## Visualisierung

Die Punktwolke ist nach der aktuellen Zuordnung eingefärbt, Sterne markieren die aktuellen
Depot-Standorte – ein Animationsschritt ist ein vollständiger Zuweisung-dann-Update-Zyklus
von Lloyd's Algorithmus, in der tatsächlichen Ausführungsreihenfolge. Ein zweites, live
mitwachsendes Diagramm zeigt die Inertia gegen die Iteration – macht sichtbar, wie schnell
das Verfahren konvergiert, nicht nur, dass es am Ende stabil wird.

Die front-and-center "📐 Wie stark hängt das Ergebnis vom Zufall der Startpunkte ab?"-
Sektion führt live 40 unabhängige Läufe je Start-Strategie auf dem aktuellen Szenario aus
und vergleicht die Verteilung der jeweils erreichten finalen Inertia – nicht nur behauptet,
sondern für jedes Szenario frisch nachgerechnet.

## Neu (2026-09-26): Mittelwert oder Medoid (k-Medoids)

Bisher wurde k-Medoids in der App nur **erklärt** (Abschnitt „Mittelwert vs. Medoid“ mit einer statischen Grafik). Jetzt lässt sich das **Zentrum** jedes Clusters umschalten: **Mittelwert** (k-Means, minimiert die Summe quadrierter Abstände) oder **Medoid** (k-Medoids, der zentralste echte Datenpunkt, minimiert die Summe der Abstände; Voronoi-Iteration). Dazu kommt ein Regler **Ausreißer** (0 bis 10 weit entfernte Punkte). Standard ist unverändert (Mittelwert, keine Ausreißer): alle bisherigen Zahlen und Presets gelten weiter, der Mittelwert-Modus liefert Schritt für Schritt bitgleich die früheren Ergebnisse (Test).
Alle Zahlen dieses Abschnitts sind in `tests/test_claims.py` belegt (Preset-Instanz: 120 Punkte, 3 Gruppen, Streuung 0,25, 5 Ausreißer, Seed 3; Verteilungen über 40 feste Netze ab Seed 100000; Werte sind das beste von 10 Läufen, Zentren-Verschiebung in Karteneinheiten).

**Ein Ausreißer zieht den Mittelwert – der Medoid bleibt.** Preset-Instanz: der Mittelwert gibt ein Zentrum an die Ausreißer ab und verschmilzt zwei echte Gruppen (Zentren im Mittel **4,87** gegenüber der Lösung ohne Ausreißer verschoben, **33,3 %** der echten Punkte falsch zugeordnet), der Medoid bleibt in den echten Gruppen (Verschiebung **0,00**, **0 %** falsch). Jedes Verfahren gewinnt in seinem eigenen Maß: Summe der Abstände 176,3 (Medoid) gegen 273,6 (Mittelwert), Inertia 1 004,5 gegen 873,7.
Über 40 Netze mit 1 / 3 / 5 / 10 Ausreißern: mittlere Verschiebung der Zentren **Mittelwert 0,106 / 0,578 / 0,836 / 3,573, Medoid 0,011 / 0,050 / 0,075 / 2,056** (Median Mittelwert 0,106 / 0,287 / 0,469 / 4,689, Medoid 0,000 / 0,000 / 0,014 / 0,220); der Medoid wird in 37 / 40 / 40 / 33 von 40 Netzen weniger verschoben. Falsch zugeordnete echte Punkte im Mittel: Mittelwert 0,1 / 1,8 / 2,7 / 21,8 %, Medoid 0,1 / 0,1 / 0,1 / 12,6 % (schlechtestes Netz bei 3 Ausreißern 33,3 % gegen 0,8 %).

**Was der Medoid kostet.** Auf sauberen Daten (40 Netze ohne Ausreißer) hat die Medoid-Lösung im Mittel die **1,031-fache Inertia** des Mittelwerts (höchstens 1,081, in allen 40 Netzen größer), der Mittelwert dagegen nur die 0,9955-fache Summe der Abstände des Medoids (er darf zwischen die Punkte gehen); beide brauchen im Mittel 1,4 Iterationen. Der Medoid-Update prüft alle Punktpaare im Cluster: O(|S|²) statt O(|S|).

**Die Medoid-Heuristik gegen den exakten p-Median.** k-Medoids ist das p-Median-Problem der Standortplanungs-Linie; die Demo löst es auf Abruf (bis 80 Punkte) exakt per MILP (HiGHS). 40 Netze mit 60 Punkten: ein Lauf mit k-Means++-Start liegt im Mittel beim **1,024-fachen** (3 Gruppen) bzw. **1,081-fachen** (5 Gruppen) des Optimums, mit Zufalls-Start 1,117 bzw. 1,097; das **beste von 40 Läufen trifft das Optimum in 40 bzw. 39 von 40 Netzen**. Die k-Means-Lösung (Mittelwert-Zentren, frei liegend) hat 0,998 bzw. 1,006 der Optimalsumme (0,976 bis 1,025).

**Was nicht wie erwartet ausfiel.**
- **„k-Means++ ist immer die bessere Start-Strategie“ – mit Ausreißern nicht.** In der Preset-Instanz mit dem Medoid liegt die mittlere Summe der Abstände bei Zufalls-Start bei 198,7, bei k-Means++ bei 254,6 (nahe am Besten in 80 % gegen 35 % der Läufe): k-Means++ wählt die weit entfernten Ausreißer bevorzugt als Startpunkte.
- **„Der Medoid rettet nicht-konvexe Formen“ – nein.** Halbmonde (k = 2): Fehlzuordnung 25,4 % (Mittelwert) gegen 23,9 % (Medoid), beide scheitern.
- **„Der Medoid ist bei vielen Ausreißern immun“ – nein.** Bei 10 Ausreißern (8 % der Punkte) verschiebt er die Zentren im Mittel immer noch um 2,06 und ordnet 12,6 % falsch zu (schlechtestes Netz 34,2 %): auch ein Medoid kann zum Ausreißer abwandern, wenn die Ausreißer eine eigene Gruppe bilden.
- **Voronoi-Iteration ist kein PAM.** Die hier gerechnete Heuristik tauscht nicht gezielt Medoide gegen andere Punkte; ein Einzellauf ist deshalb im Mittel 2 bis 12 % über dem Optimum, erst mehrere Starts nähern sich ihm.

## Sicherheitsgrenzen

`MAX_ITERATIONS` (50) begrenzt Lloyd's Algorithmus pro Lauf – bei normalen Szenariogrößen
wird das nie erreicht (Konvergenz erfolgt typischerweise nach 5–15 Iterationen), dient nur
als harte Absicherung gegen einen theoretisch möglichen Grenzfall.

## Verifikation

Ein exaktes globales Optimum ist für k-Means NP-schwer zu berechnen (Aloise et al., 2009 –
bereits für k=2, wenn die Dimension nicht beschränkt ist; in der Ebene für beliebiges k: Mahajan/Nimbhorkar/Varadarajan), es gibt also keinen zweiten, exakten Löser zum Vergleich.
Stattdessen drei unabhängige Prüfungen:

- **Monotonie-Test**: die Inertia darf über keine Iteration hinweg steigen – die
  Grundvoraussetzung für die gesamte Konvergenz-Aussage der Demo, geprüft über viele
  Zufallsinstanzen und beide Start-Strategien.
- **Handgerechnete Kleinstinstanz**: vier Punkte, zwei klar getrennte Paare. Mit
  Startzentren in der Mitte jedes Paares konvergiert das Verfahren sofort zur optimalen
  Partition (Inertia 1.0); mit beiden Startzentren im selben Paar konvergiert es stabil,
  aber zu einer nachweislich schlechteren Partition (Inertia 100.0) – der exakte Beleg
  dafür, dass Konvergenz allein keine Optimalität garantiert.
- **Kreuzvergleich mit scikit-learn**: von identischen Startzentren aus muss die eigene
  Lloyd-Implementierung dieselbe finale Inertia erreichen wie `sklearn.cluster.KMeans`
  (scikit-learn ist ausschließlich ein Test-Dependency, kein Laufzeit-Dependency der App).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Hauptablauf: Presets, Einstellungen, Iterations-Animation, Multistart-Vergleich, Formulierungs-Expander |
| `km_constants.py` | Defaults, Regler-Grenzen, Sicherheitsgrenzen, `PRESETS` |
| `km_presets.py` | `SettingSpec`/`SETTING_SPECS`, Permalink-Logik, Presets, Zufalls-Seed-Button |
| `km_scenario.py` | Zufällige Kundenpunktwolken mit einstellbarer Überlappung und Größen-Ungleichgewicht |
| `km_algorithm.py` | Lloyd's Algorithmus from scratch (Zufalls- und k-Means++-Init, Zentrum Mittelwert oder Medoid) mit vollständigem Iterations-Protokoll |
| `km_exact.py` | exakter p-Median per MILP (HiGHS) als Referenz für die Medoid-Heuristik, bis 80 Punkte |
| `km_evaluation.py` | Live-Kennzahlen pro Schritt, Multistart-Vergleich beider Start-Strategien, Mittelwert-gegen-Medoid-Vergleich und Ausreißer-Experiment |
| `km_visualization.py` | Punktwolken-, Inertia- und Verteilungs-Diagramm (Plotly) |
| `tests/` | Monotonie, Handinstanz, sklearn-Kreuzvergleich, Szenario-Reproduzierbarkeit, Multistart-Statistik, Medoid (von Hand, Brute Force, Regression des Mittelwert-Modus), Ausreißer, exakter p-Median, Zahlen (`test_claims.py`) |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Clustering erklärt: k-Means bis HDBSCAN](https://sebastianhanisch.net/konzepte-clustering.html).
