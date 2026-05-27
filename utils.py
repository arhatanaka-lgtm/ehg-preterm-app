"""
utils.py — Funções auxiliares de visualização
Goldsztejn & Nehorai, PLoS ONE 2023 — EHG App Educacional
"""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

COLORS = {
    "Clínico":           "#5C6BC0",
    "EHG":               "#26A69A",
    "Combinado":         "#EF5350",
    "Limite superior":   "#BDBDBD",
}

def plot_roc_curves(models_data: dict) -> go.Figure:
    """
    Plota curvas ROC para os modelos publicados.
    models_data: {nome: (fpr_array, tpr_array, auc, ci_lo, ci_hi)}
    """
    fig = go.Figure()

    # Linha de referência (chance)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="#BDBDBD", width=1.5, dash="dash"),
        name="Chance (AUC=0,50)",
        showlegend=True,
    ))

    order = ["Clínico", "EHG", "Combinado", "Limite superior teórico"]
    for nome, (fpr, tpr, auc, lo, hi) in models_data.items():
        key = nome.split(" (")[0]
        color = COLORS.get(key, "#78909C")
        width = 3 if key == "Combinado" else 2
        dash = "dot" if "Limite" in nome else "solid"
        ci_str = f" (IC95% {lo:.2f}–{hi:.2f})" if lo else ""
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"{nome}: AUC={auc:.2f}{ci_str}",
            line=dict(color=color, width=width, dash=dash),
            hovertemplate=f"<b>{nome}</b><br>FPR: %{{x:.2f}}<br>TPR: %{{y:.2f}}<extra></extra>",
        ))

    # Pontos da Tabela 2 (especificidade, sensibilidade)
    from models import TABLE2_POINTS
    markers = {"Clínico": "circle", "EHG": "square", "Combinado": "diamond"}
    for mod, pts in TABLE2_POINTS.items():
        color = COLORS.get(mod, "#78909C")
        fpr_pts = [1 - s for s, _ in pts]
        tpr_pts = [t for _, t in pts]
        esp_pts = [f"{int(s*100)}%" for s, _ in pts]
        fig.add_trace(go.Scatter(
            x=fpr_pts, y=tpr_pts,
            mode="markers",
            name=f"Pontos Tabela 2 — {mod}",
            marker=dict(symbol=markers.get(mod, "circle"), size=10,
                        color=color, line=dict(color="white", width=1.5)),
            text=[f"Esp={e}" for e in esp_pts],
            hovertemplate=(f"<b>{mod}</b><br>Especificidade: %{{text}}<br>"
                           f"Sensibilidade: %{{y:.1%}}<extra></extra>"),
            showlegend=False,
        ))

    fig.update_layout(
        title=dict(
            text="Curvas ROC — Predição de Parto Pré-Termo por EHG",
            font=dict(size=16, family="Arial", color="#1A237E"),
        ),
        xaxis=dict(title="Taxa de Falso Positivo (1 − Especificidade)",
                   range=[0, 1], tickformat=".0%",
                   gridcolor="#E8EAF6", linecolor="#BDBDBD"),
        yaxis=dict(title="Sensibilidade (Taxa de Verdadeiro Positivo)",
                   range=[0, 1], tickformat=".0%",
                   gridcolor="#E8EAF6", linecolor="#BDBDBD"),
        legend=dict(x=0.45, y=0.08, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#BDBDBD", borderwidth=1,
                    font=dict(size=11)),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="white",
        width=780, height=560,
        margin=dict(l=60, r=20, t=60, b=60),
        hovermode="x unified",
    )
    return fig


def plot_freq_bands(bands: dict) -> go.Figure:
    """Gráfico de barras das AUCs por banda de frequência."""
    labels = list(bands.keys())
    aucs   = list(bands.values())
    colors = ["#90CAF9", "#64B5F6", "#42A5F5", "#1E88E5", "#1A237E"]

    fig = go.Figure(go.Bar(
        x=labels, y=aucs,
        marker=dict(color=colors, line=dict(color="white", width=1.5)),
        text=[f"AUC≈{a:.2f}" for a in aucs],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>AUC ≈ %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#BDBDBD",
                  annotation_text="Chance", annotation_position="right")
    fig.update_layout(
        title=dict(text="AUC por Banda de Frequência do EHG (Fig. 3b)",
                   font=dict(size=15, color="#1A237E")),
        xaxis=dict(title="Banda de Frequência", tickangle=-10,
                   gridcolor="#E8EAF6"),
        yaxis=dict(title="AUC", range=[0.4, 0.85],
                   gridcolor="#E8EAF6", tickformat=".2f"),
        plot_bgcolor="#FAFAFA", paper_bgcolor="white",
        margin=dict(l=60, r=20, t=60, b=100),
        height=420,
        showlegend=False,
    )
    return fig


def plot_duration_auc(duration_data: dict) -> go.Figure:
    """Curva AUC × duração da gravação EHG."""
    durations = list(duration_data.keys())
    aucs      = list(duration_data.values())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=durations, y=aucs,
        mode="lines+markers",
        line=dict(color="#1A237E", width=2.5),
        marker=dict(size=9, color="#3F51B5", line=dict(color="white", width=2)),
        hovertemplate="<b>%{x} minutos</b><br>AUC = %{y:.2f}<extra></extra>",
        fill="tozeroy",
        fillcolor="rgba(26,35,126,0.08)",
    ))
    fig.add_annotation(x=1, y=0.71, text="1 min<br>AUC=0,71",
                       showarrow=True, arrowhead=2, arrowcolor="#EF5350",
                       font=dict(color="#EF5350", size=11))
    fig.add_annotation(x=30, y=0.74, text="30 min<br>AUC=0,74",
                       showarrow=True, arrowhead=2, ax=-40,
                       font=dict(color="#1A237E", size=11))
    fig.update_layout(
        title=dict(text="AUC vs. Duração da Gravação EHG (Fig. 3f)",
                   font=dict(size=15, color="#1A237E")),
        xaxis=dict(title="Duração da gravação (minutos)",
                   gridcolor="#E8EAF6"),
        yaxis=dict(title="AUC", range=[0.60, 0.80],
                   gridcolor="#E8EAF6", tickformat=".2f"),
        plot_bgcolor="#FAFAFA", paper_bgcolor="white",
        height=380,
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def plot_comparison(comp_data: list) -> go.Figure:
    """Gráfico comparativo de tecnologias (Tabela 3)."""
    names, aucs, lo_errs, hi_errs, colors_bar, texts = [], [], [], [], [], []

    for tech, desfecho, pop, auc, lo, hi, destaque in comp_data:
        label = f"{tech}<br><i style='font-size:9px'>{desfecho}</i>"
        names.append(label)
        aucs.append(auc)
        lo_errs.append(auc - lo if lo else 0)
        hi_errs.append(hi - auc if hi else 0)
        colors_bar.append("#EF5350" if destaque else "#90CAF9")
        texts.append(f"{auc:.2f}")

    fig = go.Figure(go.Bar(
        x=names, y=aucs,
        marker=dict(color=colors_bar, line=dict(color="white", width=1)),
        error_y=dict(type="data", symmetric=False,
                     array=hi_errs, arrayminus=lo_errs,
                     color="#555", thickness=1.5),
        text=texts, textposition="outside",
        hovertemplate="<b>%{x}</b><br>AUC = %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0.78, line_dash="dot", line_color="#EF5350",
                  annotation_text="Presente estudo (combinado): 0,78",
                  annotation_font_color="#EF5350")
    fig.update_layout(
        title=dict(text="Comparação com Outras Tecnologias — AUC (Tabela 3)",
                   font=dict(size=15, color="#1A237E")),
        xaxis=dict(tickangle=-30, tickfont=dict(size=9)),
        yaxis=dict(title="AUC", range=[0.4, 1.0], tickformat=".2f",
                   gridcolor="#E8EAF6"),
        plot_bgcolor="#FAFAFA", paper_bgcolor="white",
        height=520,
        margin=dict(l=60, r=20, t=80, b=160),
        showlegend=False,
    )
    return fig


def gauge_chart(score: float, label: str) -> go.Figure:
    """Gauge do escore de risco relativo."""
    color = "#4CAF50" if score < 0.35 else ("#FF9800" if score < 0.60 else "#F44336")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        title={"text": label, "font": {"size": 14}},
        number={"suffix": "%", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 35], "color": "#E8F5E9"},
                {"range": [35, 60], "color": "#FFF9C4"},
                {"range": [60, 100], "color": "#FFEBEE"},
            ],
            "threshold": {"line": {"color": "red", "width": 4},
                          "thickness": 0.75, "value": 60},
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20),
                      paper_bgcolor="white")
    return fig
