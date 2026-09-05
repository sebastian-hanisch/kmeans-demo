# k-Means für die Standortwahl von Depots – Streamlit-Demo

Zweites Stück der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research
und Machine Learning" (nach [branch-bound-demo](../branch-bound-demo)): anders als die
Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese
Demo **ein** Verfahren – k-Means (Lloyd's Algorithmus) – und lässt stattdessen die
**Optimierungslandschaft** wachsen. Vehikel-Problem: Kundenstandorte auf k Depots
aufteilen, jedes Depot bedient die ihm am nächsten liegenden Kunden – die Summe der
quadrierten Entfernungen zu minimieren ist exakt die k-Means-Zielfunktion.

Die Konzepte-Reihe ist kein linearer Pfad, sondern mehrere unabhängige Linien: diese Demo
ist der Startpunkt einer eigenen **Clustering-Linie**, unabhängig von
[branch-bound-demo](../branch-bound-demo)s Exakte-Suche-Linie. Fortsetzung dieser Linie:
[dbscan-demo](../dbscan-demo), motiviert durch zwei konkrete Schwächen von k-Means, die
DBSCAN behebt (k muss nicht vorab feststehen, Cluster müssen nicht konvex/kugelförmig
sein) – geplant danach **HDBSCAN**, motiviert durch DBSCANs eigene Schwäche bei Clustern
sehr unterschiedlicher Dichte (ein einzelnes globales `eps` reicht dann nicht mehr).

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

## Sicherheitsgrenzen

`MAX_ITERATIONS` (50) begrenzt Lloyd's Algorithmus pro Lauf – bei normalen Szenariogrößen
wird das nie erreicht (Konvergenz erfolgt typischerweise nach 5–15 Iterationen), dient nur
als harte Absicherung gegen einen theoretisch möglichen Grenzfall.

## Verifikation

Ein exaktes globales Optimum ist für k-Means NP-schwer zu berechnen (Aloise et al., 2009 –
bereits für k=2 in der Ebene), es gibt also keinen zweiten, exakten Löser zum Vergleich.
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
| `km_algorithm.py` | Lloyd's Algorithmus from scratch (Zufalls- und k-Means++-Init) mit vollständigem Iterations-Protokoll |
| `km_evaluation.py` | Live-Kennzahlen pro Schritt, Multistart-Vergleich beider Start-Strategien |
| `km_visualization.py` | Punktwolken-, Inertia- und Verteilungs-Diagramm (Plotly) |
| `tests/` | Monotonie, Handinstanz, sklearn-Kreuzvergleich, Szenario-Reproduzierbarkeit, Multistart-Statistik |

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

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
