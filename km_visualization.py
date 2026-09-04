"""Plotly-Visualisierungen: Punktwolke mit Clusterzuordnung (Kernvisual, Schritt-fuer-
Schritt animierbar wie der Suchbaum in branch-bound-demo), Inertia-Konvergenz-Diagramm
und die Verteilung der finalen Inertia ueber die Multistart-Vergleichslaeufe."""

import numpy as np

CLUSTER_PALETTE = [
    "#1f77b4", "#d68a2e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]


def build_scatter_figure(instance, result, step):
    import plotly.graph_objects as go

    data = np.array(instance.points)
    s = result.steps[step]
    labels = np.array(s.labels)
    centers = np.array(s.centers)
    k = centers.shape[0]

    fig = go.Figure()
    for i in range(k):
        mask = labels == i
        color = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=data[mask, 0], y=data[mask, 1], mode="markers", name=f"Cluster {i + 1}",
                marker=dict(color=color, size=7, line=dict(width=0.5, color="white")),
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=centers[:, 0], y=centers[:, 1], mode="markers", name="Zentren",
            marker=dict(
                symbol="star", size=18,
                color=[CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i in range(k)],
                line=dict(width=1.5, color="#14233B"),
            ),
            hoverinfo="skip",
        )
    )

    xmin, xmax = data[:, 0].min(), data[:, 0].max()
    ymin, ymax = data[:, 1].min(), data[:, 1].max()
    padx = (xmax - xmin) * 0.1 or 1.0
    pady = (ymax - ymin) * 0.1 or 1.0
    fig.update_layout(
        template="plotly_white", height=460,
        xaxis=dict(visible=False, range=[xmin - padx, xmax + padx], fixedrange=True),
        yaxis=dict(visible=False, range=[ymin - pady, ymax + pady], fixedrange=True, scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=40, l=10, r=10, b=10),
    )
    return fig


def build_inertia_chart(result, step):
    import plotly.graph_objects as go

    xs = [s.iteration for s in result.steps if s.iteration <= step]
    ys = [s.inertia for s in result.steps if s.iteration <= step]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color="#2ca02c", width=3),
                    marker=dict(size=6), name="Inertia")
    )
    fig.update_layout(
        template="plotly_white", height=280,
        xaxis_title="Iteration", yaxis_title="Inertia (Summe quadrierter Abstaende)",
        showlegend=False, margin=dict(t=20, l=10, r=10, b=10),
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_multistart_distribution_chart(comparison):
    import plotly.graph_objects as go

    fig = go.Figure()
    for key, label, color in [
        ("random", "Zufaellige Startpunkte", "#d68a2e"),
        ("kmeans++", "k-Means++", "#1f77b4"),
    ]:
        summary = comparison.strategies[key]
        fig.add_trace(
            go.Box(
                y=list(summary.final_inertias), name=label, marker_color=color,
                boxpoints="all", jitter=0.4, pointpos=0,
            )
        )
    fig.add_hline(
        y=comparison.global_best_inertia, line=dict(color="#14233B", width=1.5, dash="dash"),
        annotation_text="Bestes gefundenes Ergebnis", annotation_position="top left",
    )
    fig.update_layout(
        template="plotly_white", height=320, yaxis_title="Finale Inertia nach Konvergenz",
        showlegend=False, margin=dict(t=30, l=10, r=10, b=10),
    )
    fig.update_yaxes(fixedrange=True)
    return fig
