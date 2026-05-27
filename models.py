"""
models.py — Dados e lógica dos modelos do artigo EHG
Goldsztejn U, Nehorai A. PLoS ONE. 2023;18(5):e0285219.
"""
import numpy as np
from scipy import stats

# ─── AUCs publicados ────────────────────────────────────────────────────────────────────────────
AUCS = {
    "Clínico (Classificação)": (0.65, 0.63, 0.67),
    "EHG (Classificação)":     (0.74, 0.73, 0.76),
    "Combinado (Classificação)":(0.78, 0.76, 0.80),
    "Clínico (Regressão)":     (0.67, 0.65, 0.70),
    "EHG (Regressão)":         (0.70, 0.68, 0.73),
    "Combinado (Regressão)":   (0.75, 0.73, 0.77),
    "Limite superior teórico": (0.98, 0.98, 0.98),
}

# ─── Pontos da tabela 2 (especificidade, sensibilidade) por modelo ────────────
TABLE2_POINTS = {
    "Clínico":   [(0.50, 0.653), (0.70, 0.520), (0.90, 0.273)],
    "EHG":       [(0.50, 0.847), (0.70, 0.647), (0.90, 0.348)],
    "Combinado": [(0.50, 0.873), (0.70, 0.722), (0.90, 0.397)],
}

# ─── PPV/NPV por modelo e especificidade ───────────────────────────────────────────────────
PERFORMANCE_TABLE = {
    "Sensibilidade": {
        "Clínico":   {"50%": "65,3% (61,8–68,9)", "70%": "52,0% (48,4–55,6)", "90%": "27,3% (23,1–31,5)"},
        "EHG":       {"50%": "84,7% (81,5–86,5)", "70%": "64,7% (61,0–68,4)", "90%": "34,8% (31,0–38,6)"},
        "Combinado": {"50%": "87,3% (84,3–90,3)", "70%": "72,2% (68,6–75,8)", "90%": "39,7% (35,7–43,6)"},
    },
    "PPV": {
        "Clínico":   {"50%": "23,3% (22,3–24,3)", "70%": "28,7% (27,2–30,3)", "90%": "38,9% (34,9–42,8)"},
        "EHG":       {"50%": "28,1% (27,5–28,7)", "70%": "33,4% (31,9–34,8)", "90%": "44,8% (41,6–47,9)"},
        "Combinado": {"50%": "28,9% (28,1–29,6)", "70%": "35,9% (34,6–37,2)", "90%": "48,0% (45,0–51,0)"},
    },
    "NPV": {
        "Clínico":   {"50%": "86,1% (84,9–87,4)", "70%": "86,2% (85,4–87,1)", "90%": "84,2% (83,4–85,0)"},
        "EHG":       {"50%": "93,1% (92,1–94,1)", "70%": "89,5% (88,5–90,5)", "90%": "85,6% (84,9–86,3)"},
        "Combinado": {"50%": "94,4% (93,2–95,6)", "70%": "91,5% (90,6–92,5)", "90%": "86,5% (85,8–87,3)"},
    },
}

# ─── Análise de bandas de frequência (estimativa a partir da Fig. 3b) ─────────
FREQ_BANDS = {
    "B0 (0,05–1,0 Hz)
Contrações uterinas": 0.58,
    "B1 (1,0–2,2 Hz)
Harmônicos baixos":    0.63,
    "B2 (2,2–3,5 Hz)
Reverbração cardíaca": 0.69,
    "B3 (3,5–5,0 Hz)
Harmônicos altos":     0.72,
    "Full (0,05–4,0 Hz)
Espectro completo":  0.74,
}

# ─── Duração da gravação vs AUC (estimativa a partir da Fig. 3f) ─────────────
DURATION_AUC = {
    1:  0.71,
    3:  0.71,
    5:  0.72,
    10: 0.72,
    15: 0.73,
    20: 0.73,
    25: 0.74,
    30: 0.74,
}

# ─── Comparação com outras tecnologias (Tabela 3) ────────────────────────────────────────────
COMPARISON_DATA = [
    # (Tecnologia, Desfecho, População, AUC_centro, AUC_min, AUC_max, Destaque)
    ("Informações clínicas",          "< 37 sem", "Gestação única, assintomáticas", 0.64, 0.61, 0.67, False),
    ("Modelo Clínico (Presente estudo)","< 37 sem","TPEHGDB+TPEHGTDS",             0.65, 0.63, 0.67, False),
    ("Fibronectina alfa",             "< 37 sem", "Assintomáticas",                0.65, 0.63, 0.66, False),
    ("Fibronectina alfa",             "< 37 sem", "Sintomáticas",                  0.71, 0.69, 0.73, False),
    ("Comprimento colo uterino",      "< 37 sem", "Nulíparas, assintomáticas",     0.67, 0.64, 0.70, False),
    ("Clínico + Placentário",         "< 37 sem", "Gestação única, assintomáticas",0.72, 0.66, 0.77, False),
    ("EHG (outros métodos)",          "< 37 sem", "TPEHGDB",                       0.61, None, None, False),
    ("EHG end-to-end DL",             "< 37 sem", "TPEHGDB",                       0.69, None, None, False),
    ("EHG (Presente estudo) ★",       "< 37 sem", "TPEHGDB+TPEHGTDS",             0.74, 0.73, 0.76, True),
    ("Combinado EHG+Clínico (Presente estudo) ★★","< 37 sem","TPEHGDB+TPEHGTDS", 0.78, 0.76, 0.80, True),
    ("Fibronectina alfa",             "< 1 sem",  "Sintomáticas (imminente)",      0.84, 0.80, 0.87, False),
    ("Comprimento colo uterino",      "< 1 sem",  "Sintomáticas (imminente)",      0.84, None, None, False),
]

# ─── Variáveis clínicas com risco relativo ───────────────────────────────────────────────
CLINICAL_VARS = [
    # (nome_exibição, nome_interno, tipo, RR_centro, RR_min, RR_max)
    ("Paridade (multípara)", "parous",   "bin", 1.32, 0.79, 2.20),
    ("Aborto prévio",        "abortion", "bin", 0.72, 0.23, 2.28),
    ("Sangramento 1T",       "bleed1",   "bin", 0.25, 0.02, 4.13),
    ("Sangramento 2T",       "bleed2",   "bin", 12.53, 1.19, 131.49),
    ("Tabagismo",            "smoker",   "bin", 3.58, 1.17, 10.96),
]


def binormal_roc(auc: float, n_points: int = 200):
    """
    Gera curva ROC aproximada usando modelo binormal simétrico.
    TPR = Φ(a + b·Φ⁻¹(FPR)), onde a = √2 · Φ⁻¹(AUC), b = 1.
    """
    a = np.sqrt(2) * stats.norm.ppf(auc)
    fpr = np.linspace(0.001, 0.999, n_points)
    tpr = stats.norm.cdf(a + stats.norm.ppf(fpr))
    fpr = np.concatenate([[0], fpr, [1]])
    tpr = np.concatenate([[0], tpr, [1]])
    return fpr, tpr


def compute_risk_score(inputs: dict) -> dict:
    """
    Calcula escore de risco relativo aproximado com base nos log(RR)
    das variáveis binárias publicadas. Retorna score normalizado [0–1]
    e classificação.
    NOTA: Os coeficientes β do LASSO não foram publicados.
    Escore é APROXIMADO e de uso exclusivamente educacional.
    """
    log_rr_sum = 0.0
    log_rr_max = sum(abs(np.log(v[3])) for v in CLINICAL_VARS)

    for _, nome, tipo, rr, _, _ in CLINICAL_VARS:
        if inputs.get(nome, False):
            log_rr_sum += np.log(rr)

    # Normaliza entre -1 e +1 → converte para [0,1]
    score = (log_rr_sum / log_rr_max + 1) / 2 if log_rr_max > 0 else 0.5
    score = max(0.05, min(0.95, score))

    if score < 0.35:
        nivel = "🟢 Baixo risco relativo"
    elif score < 0.60:
        nivel = "🟡 Risco intermediário"
    else:
        nivel = "🔴 Alto risco relativo"

    return {"score": score, "nivel": nivel}
