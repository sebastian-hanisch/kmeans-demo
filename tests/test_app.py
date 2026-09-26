"""End-to-end Smoke-Test via Streamlits offizielles AppTest-Framework: laedt app.py mit
den Standardeinstellungen und prueft, dass kein Python-Fehler auftritt. Ergaenzt die
funktionalen Unit-Tests der uebrigen Module - ein Fehler wie
`streamlit.errors.StreamlitDuplicateElementId` (zwei st.plotly_chart-Aufrufe ohne
eindeutiges key= rendern zufaellig identischen Inhalt und kollidieren) liegt in app.py's
Widget-Verdrahtung selbst und kann nur durch einen echten End-to-End-Lauf gefunden werden,
nicht durch Unit-Tests der Algorithmus-/Visualisierungs-Module (siehe hdbscan-demo, wo
genau dieser Fehlertyp bei einem echten Nutzer auftrat, 2026-09-05)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import km_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def test_app_loads_without_exception():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]


@pytest.mark.parametrize("preset_name", list(C.PRESETS.keys()))
def test_all_presets_load_without_exception(preset_name):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=120)
    buttons = {b.label: b for b in at.button}
    buttons[preset_name].click().run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]


def _run(setup=None):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=180)
    assert not at.exception, [str(e) for e in at.exception]
    if setup is not None:
        setup(at)
        at.run(timeout=180)
        assert not at.exception, [str(e) for e in at.exception]
    return at


def _metrics(at, label):
    return [m.value for m in at.metric if m.label == label]


def _click_preset(name):
    def setup(at):
        {b.label: b for b in at.button}[name].click()
    return setup


@pytest.mark.parametrize("center", C.CENTER_TYPES)
@pytest.mark.parametrize("preset_name", list(C.PRESETS.keys()))
def test_every_preset_renders_with_both_centers(preset_name, center):
    at = _run(_click_preset(preset_name))
    at.session_state["center_radio"] = center            # nach dem Preset (das eigene Zentrum setzt) umstellen
    at.run(timeout=180)
    assert not at.exception, [str(e) for e in at.exception]
    assert _metrics(at, C.OBJECTIVE_LABELS[center])


@pytest.mark.parametrize("center", C.CENTER_TYPES)
@pytest.mark.parametrize("n_out", [C.OUTLIERS_MIN, C.OUTLIERS_MAX])
def test_outlier_extremes_render(center, n_out):
    def setup(at):
        at.session_state["center_radio"] = center
        at.session_state["outliers_slider"] = n_out
    at = _run(setup)
    assert not at.error
    assert any("Stellen Sie links" in i.value for i in at.info) == (n_out == 0)


def test_default_still_shows_the_inertia_label_and_the_old_number():
    at = _run()
    assert _metrics(at, "Inertia (WCSS)") == ["213.6"]


def test_outlier_presets_show_the_documented_effect():
    at = _run(_click_preset("Ausreißer ziehen den Mittelwert"))
    assert any("33.3 %" in str(t.value.values.tolist()) for t in at.table)
    at = _run(_click_preset("Medoid hält gegen Ausreißer"))
    assert _metrics(at, "Summe der Abstände") and any("0.0 %" in str(t.value.values.tolist()) for t in at.table)


def test_permalink_center_and_outliers_are_loaded_and_clamped():
    at = AppTest.from_file(APP_PATH)
    at.query_params["center"] = "medoid"
    at.query_params["out"] = "99"
    at.run(timeout=180)
    assert not at.exception
    assert at.sidebar.radio(key="center_radio").value == "medoid" and at.sidebar.slider(key="outliers_slider").value == C.OUTLIERS_MAX


def test_invalid_permalink_center_falls_back_to_the_default():
    at = AppTest.from_file(APP_PATH)
    at.query_params["center"] = "median"
    at.run(timeout=180)
    assert not at.exception and at.sidebar.radio(key="center_radio").value == C.DEFAULT_CENTER


def test_exact_button_is_disabled_above_the_limit_and_works_below():
    at = _run()
    assert next(b for b in at.button if b.key == "exact_start").disabled                    # Standard: 120 Punkte > 80
    at = _run(lambda a: a.session_state.__setitem__("n_points_slider", 40))
    btn = next(b for b in at.button if b.key == "exact_start")
    assert not btn.disabled
    btn.click().run(timeout=180)
    assert not at.exception and any("Exakt (p-Median" in str(t.value.values.tolist()) for t in at.table)


def test_outlier_experiment_runs_on_demand(monkeypatch):
    import km_evaluation as ev
    orig = ev.outlier_experiment
    monkeypatch.setattr(ev, "outlier_experiment", lambda *a, **kw: orig(*a, seeds=C.EXPERIMENT_SEEDS[:2], counts=(1, 3), **kw))
    at = _run(lambda a: a.session_state.__setitem__("spread_slider", 0.3))
    next(b for b in at.button if b.key == "outlier_exp_start").click().run(timeout=180)
    assert not at.exception and any("Medoid weniger verschoben in" in str(t.value.columns.tolist()) for t in at.table)
