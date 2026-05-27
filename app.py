"""
app.py — EHG + Deep Learning na Predição do Parto Pré-Termo
Dashboard educacional interativo

Referência:
Goldsztejn U, Nehorai A. Predicting preterm births from electrohysterogram
recordings via deep learning. PLoS ONE. 2023;18(5):e0285219.
DOI: https://doi.org/10.1371/journal.pone.0285219

Desenvolvido para uso no Congresso Brasileiro de Ginecologia e Obstetrícia.
FINALIDADE EXCLUSIVAMENTE EDUCACIONAL — Não substitui julgamento clínico.
"""

import streamlit as st
import pandas as pd
import numpy as np

from models import (
    AUCS, TABLE2_POINTS, PERFORMANCE_TABLE,
    FREQ_BANDS, DURATION_AUC, COMPARISON_DATA,
    CLINICAL_VARS, binormal_roc, compute_risk_score,
)
from utils import (
    plot_roc_curves, plot_freq_bands, plot_duration_auc,
    plot_comparison, gauge_chart,
)

# ─── Configuração da página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="EHG + IA na Predição do Parto Pré-Termo",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS customizado ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1A237E 0%, #3F51B5 100%);
        color: white;
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.7rem; }
    .main-header p  { color: #C5CAE9; margin: 6px 0 0; font-size: 0.9rem; }
    .metric-card {
        background: white;
        border: 1px solid #E8EAF6;
        border-left: 5px solid #3F51B5;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 6px 0;
    }
    .metric-card h3 { color: #1A237E; margin: 0; font-size: 1.4rem; }
    .metric-card p  { color: #555; margin: 4px 0 0; font-size: 0.85rem; }
    .highlight-card {
        background: #E8F5E9;
        border: 1px solid #A5D6A7;
        border-left: 5px solid #4CAF50;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 6px 0;
    }
    .highlight-card h3 { color: #2E7D32; margin: 0; font-size: 1.4rem; }
    .highlight-card p  { color: #555; margin: 4px 0 0; font-size: 0.85rem; }
    .warning-box {
        background: #FFF9C4;
        border: 1px solid #F9A825;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 0.88rem;
    }
    .info-box {
        background: #E3F2FD;
        border: 1px solid #90CAF9;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 0.88rem;
    }
    .section-title {
        color: #1A237E;
        border-bottom: 2px solid #3F51B5;
        padding-bottom: 6px;
        margin-top: 20px;
    }
    div[data-testid="stTabs"] button {
        font-size: 0.9rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ─── Cabeçalho principal ──────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🫀 EHG + Inteligência Artificial na Predição do Parto Pré-Termo</h1>
    <p>
        Goldsztejn U, Nehorai A. <b>Predicting preterm births from electrohysterogram recordings
        via deep learning.</b> <i>PLoS ONE</i>. 2023;18(5):e0285219 &nbsp;|&nbsp;
        DOI: <a href="https://doi.org/10.1371/journal.pone.0285219"
        style="color:#90CAF9">10.1371/journal.pone.0285219</a>
    </p>
    <p>🏛️ Congresso Brasileiro de Ginecologia e Obstetrícia &nbsp;|&nbsp;
       ⚠️ Ferramenta exclusivamente educacional — não substitui julgamento clínico</p>
</div>
""", unsafe_allow_html=True)

# ─── Abas principais ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Curvas ROC",
    "📈 Performance Detalhada",
    "🔬 Análise de Bandas EHG",
    "🌐 Comparação Global",
    "👩‍⚕️ Risco Clínico (Aprox.)",
    "📚 Como Funciona",
])

# ══════════════════════════════════════════════════════════════════════════════
# ABA 1 — CURVAS ROC
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<h2 class="section-title">Curvas ROC dos Modelos</h2>', unsafe_allow_html=True)

    # Gera curvas ROC aproximadas via modelo binormal
    models_roc = {}
    for nome, (auc, lo, hi) in AUCS.items():
        fpr, tpr = binormal_roc(auc)
        models_roc[nome] = (fpr, tpr, auc, lo, hi)

    col_plot, col_metrics = st.columns([3, 1])
    with col_plot:
        fig_roc = plot_roc_curves(models_roc)
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_metrics:
        st.markdown("#### AUC dos Modelos")
        data_metrics = [
            ("Clínico", "0,65", "(0,63–0,67)", "#5C6BC0", False),
            ("EHG (Deep Learning)", "0,74", "(0,73–0,76)", "#26A69A", False),
            ("Combinado ★", "0,78", "(0,76–0,80)", "#EF5350", True),
            ("Regressão Clínico", "0,67", "(0,65–0,70)", "#7986CB", False),
            ("Regressão EHG", "0,70", "(0,68–0,73)", "#4DB6AC", False),
            ("Regressão Combinado", "0,75", "(0,73–0,77)", "#FF7043", False),
            ("Limite teórico máx.", "0,98", "(0,98–0,98)", "#BDBDBD", False),
        ]
        for nome, auc_val, ci, cor, destaque in data_metrics:
            card_class = "highlight-card" if destaque else "metric-card"
            st.markdown(f"""
            <div class="{card_class}" style="border-left-color:{cor}">
                <h3>{auc_val}</h3>
                <p><b>{nome}</b><br>{ci}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    📌 <b>Como ler o gráfico:</b> Os pontos marcados correspondem aos 3 limiares da Tabela 2
    (especificidade 50%, 70% e 90%). As curvas são aproximações pelo modelo binormal com os
    AUCs publicados. O limite teórico máximo (AUC=0,98) reflete a incerteza inerente da
    estimativa da idade gestacional (DP≈6 dias).
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ABA 2 — PERFORMANCE DETALHADA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<h2 class="section-title">Performance por Limiar — Tabela 2</h2>',
                unsafe_allow_html=True)

    especificidade = st.radio(
        "Selecione o limiar de especificidade:",
        ["50%", "70%", "90%"],
        horizontal=True,
        index=1,
    )

    col_a, col_b, col_c = st.columns(3)
    modelos_cols = [("Clínico", col_a, "#5C6BC0"),
                    ("EHG", col_b, "#26A69A"),
                    ("Combinado", col_c, "#EF5350")]

    metricas_order = ["Sensibilidade", "PPV", "NPV"]
    for mod, col, cor in modelos_cols:
        with col:
            st.markdown(f"### {mod}")
            for met in metricas_order:
                val = PERFORMANCE_TABLE[met][mod][especificidade]
                st.markdown(f"""
                <div class="metric-card" style="border-left-color:{cor}">
                    <h3>{val.split(' ')[0]}</h3>
                    <p><b>{met}</b><br><small>{' '.join(val.split(' ')[1:])}</small></p>
                </div>""", unsafe_allow_html=True)

    st.markdown("---")
    # Tabela completa
    st.markdown("#### Tabela completa de performance (Tabela 2, Goldsztejn & Nehorai 2023)")
    rows = []
    for met in ["Sensibilidade", "PPV", "NPV"]:
        for mod in ["Clínico", "EHG", "Combinado"]:
            rows.append({
                "Métrica": met,
                "Modelo": mod,
                "Especificidade 50%": PERFORMANCE_TABLE[met][mod]["50%"],
                "Especificidade 70%": PERFORMANCE_TABLE[met][mod]["70%"],
                "Especificidade 90%": PERFORMANCE_TABLE[met][mod]["90%"],
            })
    df_perf = pd.DataFrame(rows)

    def highlight_combinado(row):
        if row["Modelo"] == "Combinado":
            return ["background-color: #E8F5E9; font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_perf.style.apply(highlight_combinado, axis=1),
        use_container_width=True,
        hide_index=True,
        height=380,
    )

    st.markdown("""
    <div class="warning-box">
    ⚠️ <b>Importante:</b> PPV e NPV dependem da prevalência de pré-termo na população.
    No dataset (TPEHGDB+TPEHGTDS), a prevalência é 18,9% — maior que a população geral (~10%).
    Na prática clínica com prevalência de 10%, o PPV será menor e o NPV será maior que os valores acima.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ABA 3 — ANÁLISE DE BANDAS EHG
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<h2 class="section-title">Análise das Componentes Preditivas do EHG</h2>',
                unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### AUC por Banda de Frequência (Fig. 3b)")
        fig_bands = plot_freq_bands(FREQ_BANDS)
        st.plotly_chart(fig_bands, use_container_width=True)

        st.markdown("""
        <div class="info-box">
        🔬 <b>Achado principal:</b> As bandas de <b>alta frequência (B2 e B3)</b> são
        <b>mais preditivas</b> de parto pré-termo do que a banda de contração uterina (B0).
        Isso sugere que harmônicos espectrais — não apenas as contrações visíveis —
        carregam informação prognóstica relevante.
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("#### AUC vs. Duração da Gravação (Fig. 3f)")
        fig_dur = plot_duration_auc(DURATION_AUC)
        st.plotly_chart(fig_dur, use_container_width=True)

        st.markdown("""
        <div class="info-box">
        ⏱️ <b>Achado clínico relevante:</b> Com apenas <b>1 minuto</b> de gravação EHG,
        o modelo mantém AUC=0,71 vs. 0,74 com 30 minutos completos.
        Isso viabiliza a implementação clínica e domiciliar com dispositivos portáteis.
        </div>
        """, unsafe_allow_html=True)

    # Padrões temporais
    st.markdown("---")
    st.markdown("#### Impacto da Disrupção dos Padrões Temporais (Fig. 3d)")
    col_ta, col_tb = st.columns([2, 1])
    with col_ta:
        frac = st.slider(
            "Fração de colunas STFT aleatorizadas (disrupção temporal):",
            0, 100, 0, step=10,
            format="%d%%",
        )
        # AUC interpolado: mínimo impacto — de 0.74 a ~0.74 (quase constante)
        auc_disrupted = 0.74 - (frac / 100) * 0.001
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:#1A237E">
            <h3>AUC ≈ {auc_disrupted:.2f}</h3>
            <p>Com {frac}% dos padrões temporais destruídos —
            {'<b>sem impacto significativo!</b>' if frac >= 80 else 'impacto mínimo'}</p>
        </div>""", unsafe_allow_html=True)

    with col_tb:
        st.markdown("""
        <div class="warning-box">
        <b>Interpretação:</b><br>
        Mesmo com 100% dos padrões temporais embaralhados (STFT completamente aleatório no tempo),
        o modelo manteve AUC=0,74. <br><br>
        <b>Conclusão:</b> É a <u>composição espectral</u> — não os padrões temporais —
        que prediz o parto pré-termo. O EHG como "impressão digital espectral" do útero.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ABA 4 — COMPARAÇÃO GLOBAL
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<h2 class="section-title">Comparação com Outras Tecnologias (Tabela 3)</h2>',
                unsafe_allow_html=True)

    fig_comp = plot_comparison(COMPARISON_DATA)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")
    # Tabela detalhada
    st.markdown("#### Tabela detalhada")
    df_comp = pd.DataFrame([
        {
            "Tecnologia": t,
            "Desfecho": d,
            "População": p,
            "AUC": f"{auc:.2f}",
            "IC 95%": f"({lo:.2f}–{hi:.2f})" if lo else "—",
            "Destaque": "★" if dest else "",
        }
        for t, d, p, auc, lo, hi, dest in COMPARISON_DATA
    ])

    def style_comp(row):
        if row["Destaque"] == "★":
            return ["background-color: #E8F5E9; font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_comp.style.apply(style_comp, axis=1),
        use_container_width=True,
        hide_index=True,
        height=420,
    )

    st.markdown("""
    <div class="info-box">
    📌 <b>Contexto importante:</b> O modelo combinado (AUC=0,78) supera a fibronectina alfa e o
    comprimento cervical na predição de pré-termo &lt;37 semanas em pacientes assintomáticas.
    Porém, esses métodos clínicos com AUC=0,84 são aplicados em pacientes com <b>sintomas iminentes</b>
    (trabalho de parto pré-termo ativo) — população completamente diferente. A comparação é
    metodologicamente diferente, não diretamente equivalente.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ABA 5 — CALCULADORA DE RISCO CLÍNICO
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<h2 class="section-title">Escore de Risco Clínico — Variáveis do Modelo M1</h2>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box">
    ⚠️ <b>AVISO IMPORTANTE:</b> Os coeficientes β do modelo de regressão logística com LASSO
    <b>não foram publicados</b> no artigo. Este escore usa os Risk Ratios (RR) brutos publicados
    como <b>aproximação</b> do risco relativo. <b>Não representa probabilidade absoluta.</b><br>
    Finalidade exclusivamente educacional/didática. AUC do modelo clínico publicado: <b>0,65</b>.
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown("#### Insira as variáveis da paciente:")

        inputs = {}

        with st.expander("📋 Informações contínuas (descritivas)"):
            idade = st.number_input("Idade materna (anos)", 15, 50, 28)
            ig_rec = st.number_input("IG na gravação (semanas)", 26, 36, 31)
            peso = st.number_input("Peso (kg)", 40, 150, 72)

        st.markdown("##### Fatores de risco (variáveis binárias):")
        inputs["parous"]   = st.checkbox("✅ Multípara (pariu anteriormente)")
        inputs["abortion"] = st.checkbox("🔄 Aborto prévio")
        inputs["bleed1"]   = st.checkbox("🩸 Sangramento no 1º trimestre")
        inputs["bleed2"]   = st.checkbox("🩸 Sangramento no 2º trimestre", value=False)
        inputs["smoker"]   = st.checkbox("🚬 Tabagista ativa")

        st.markdown("""
        <div class="info-box" style="font-size:0.82rem">
        <b>Não incluídas no modelo:</b> HAS, DM, funil cervical, posição placentária
        (excluídas por baixa prevalência ou collinearidade no dataset).
        </div>
        """, unsafe_allow_html=True)

    with col_result:
        result = compute_risk_score(inputs)
        score  = result["score"]
        nivel  = result["nivel"]

        st.markdown("#### Resultado do Escore:")
        fig_gauge = gauge_chart(score, "Escore de Risco Relativo (Aprox.)")
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        <div class="{'highlight-card' if score < 0.35 else 'metric-card'}"
             style="border-left-color:{'#4CAF50' if score < 0.35 else '#FF9800' if score < 0.60 else '#F44336'}">
            <h3>{nivel}</h3>
            <p>Escore: {score*100:.1f}% | IG na gravação: {ig_rec} sem | Peso: {peso} kg | Idade: {idade} anos</p>
        </div>""", unsafe_allow_html=True)

        # Tabela de risk ratios
        st.markdown("##### Risk Ratios publicados (Tabela 1):")
        df_rr = pd.DataFrame([
            {
                "Variável": nome,
                "Status": "✅ Presente" if inputs[chave] else "❌ Ausente",
                "RR (IC 95%)": f"{rr:.2f} ({lo:.2f}–{hi:.2f})",
                "Contribuição": "↑ Risco" if (inputs[chave] and rr > 1) or (not inputs[chave] and rr < 1) else "↓ Risco" if inputs[chave] else "—",
            }
            for nome, chave, _, rr, lo, hi in CLINICAL_VARS
        ])
        st.dataframe(df_rr, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# ABA 6 — COMO FUNCIONA
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<h2 class="section-title">Como Funciona — O Modelo de Deep Learning EHG</h2>',
                unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("""
        ### O que é o EHG?
        O **eletero-histerograma (EHG)** mede a atividade elétrica do útero por meio de
        eletrodos abdominais — semelhante a um ECG, mas para o útero. Ele captura os
        potenciais de ação das células miometriais que geram as contrações uterinas.

        O canal utilizado neste estudo (**s1**) mede a diferença de potencial entre dois
        eletrodos horizontais localizados 3,5 cm acima do umbigo, separados por 7 cm.

        ---
        ### Pipeline do Modelo (passo a passo)

        | Etapa | Operação | Parâmetros |
        |-------|----------|------------|
        | 1 | Remover 1º minuto | Efeitos transientes |
        | 2 | Filtro Butterworth 4ª ordem | 0,05–4,0 Hz, fase zero |
        | 3 | Downsampling | 10 Hz |
        | 4 | **STFT** (Transformada de Fourier de Tempo Curto) | Janela Hamming 60s, 75% sobreposição |
        | 5 | **BiLSTM** (100 células, bidirecional) | Aprende padrões espectrais ao longo do tempo |
        | 6 | Camada fully connected | 2 neurônios |
        | 7 | Softmax | Probabilidade pré-termo vs. a termo |

        ---
        ### Por que BiLSTM e não CNN?
        O BiLSTM foi escolhido porque:
        - Processa sequências temporais de forma bidirecional (passado e futuro)
        - As colunas do STFT formam uma sequência temporal de "snapshots" espectrais
        - Captura como o espectro do EHG muda ao longo dos 30 minutos de gravação
        """)

    with col_r:
        st.markdown("""
        ### Como o Modelo Combinado Funciona?

        ```
        ┌─────────────────────────────────────┐
        │  Sinal EHG (30 min, canal s1)       │
        └──────────────┬──────────────────────┘
                       │        Pré-processamento
                         ▼
                     [BPF + STFT]
                         │
                         ▼
               [BiLSTM — 100 hidden]
                          │
                          ▼
              [Fully Connected —  2 neurônios]
                         │
              (ativações extraídas)
                         │)
            ┌──────────────┐
            │  + Variáveis clínicas tabulares
            │     (idade, peso, paridade, etc.)
            └──────────────┬──────────────────────┐
                         ▼
             [Regressão Logística (LASSO)]
                          │
                          ▼
            Probabilidade de Pré-Termo
        ```

        ---
        ### Por que a especificidade espectral importa?

        O achado mais revelador do estudo: **os padrões temporais não são cruciais**.
        Quando os autores embaralharam 100% das colunas do STFT (destruindo toda informação
        temporal), o AUC pTFT intacto.

        Isso sugere que o útero de gestantes que pararão pré-termo tem uma **"assinatura espectral"**
        distinta — uma composição frequencial diferente — independentemente de quando as contrações
        ocorrem durante a gravação.

        ---
        ### Dados e Código

        - 🗄️ Datasets: [TPEHGDB](https://physionet.org/content/tpehgdb/) e
          [TPEHGTDS](https://physionet.org/content/tpehgtds/) — disponíveis no PhysioNet
        - 💻 Código: [github.com/uri-goldsztejn/Predicting_preterm_birth_from_EHG](https://github.com/uri-goldsztejn/Predicting_preterm_birth_from_EHG)
        - 🔬 Implementação: MATLAB 2020a
        """)

    # Diagrama ASCII com cores
    st.markdown("---")
    st.markdown("### Validação: 5-Fold Cross-Validation × 20 repetições")
    st.markdown("""
    Para garantir robustez estatística e evitar resultados otimistas por sorte na partição:
    - **5 folds**: 80% treino / 20% teste em cada iteração
    - **Repetido 20 vezes** com diferentes partições aleatórias
    - Total: **100 avaliações** → média ± IC 95% (distribuição Gaussiana com variância desconhecida)
    - **Prevenção de data leakage**: médias e modas para imputação calculadas **apenas** no treino
    """)

    # Rodapé
    st.markdown("---")
    st.markdown("""
    <div class="warning-box" style="text-align:center">
    <b>Citação completa (Vancouver):</b><br>
    Goldsztejn U, Nehorai A. Predicting preterm births from electrohysterogram recordings via deep learning.
    <i>PLoS ONE</i>. 2023;18(5):e0285219. doi: 10.1371/journal.pone.0285219. PMID: 37167285.<br><br>
    App desenvolvido para fins educacionais no Congresso Brasileiro de Ginecologia e Obstetrícia (FEBRASGO).<br>
    <b>Não substitui avaliação clínica individualizada.</b>
    </div>
    """, unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🫀 EHG + IA no Pré-Termo")
    st.markdown("""
    **Estudo:** Goldsztejn & Nehorai, 2023
    **Periódico:** PLoS ONE
    **DOI:** [10.1371/journal.pone.0285219](https://doi.org/10.1371/journal.pone.0285219)
    """)

    st.markdown("---")
    st.markdown("#### Principais Achados")
    st.metric("AUC Modelo Combinado", "0,78", "(IC95% 0,76–0,80)")
    st.metric("Sensibilidade @ Esp.70%", "72,2%", "IC: 68,6–75,8%")
    st.metric("NPV @ Esp.70%", "91,5%", "IC: 90,6–92,5%")
    st.metric("N da coorte", "159 gestantes", "18,9% pré-termo")

    st.markdown("---")
    st.markdown("#### Pontos-chave da Apresentação")
    st.markdown("""
    1. 🎯 **EHG supera** fibronectina e comprimento cervical em predição de pré-termo <37 sem em assintomáticas
    2. ⚡ **1 minuto** de gravação é suficiente (AUC=0,71)
    3. 📡 **Espectro > tempo**: frequência importa mais que contrações
    4. 🤖 **Sem handcrafted features**: aprendizado automático
    5. ⚠️ **Data leakage** corrigido (problema histórico da área)
    """)

    st.markdown("---")
    st.markdown("#### Limitações")
    st.markdown("""
    - Dataset único (Ljubljana, Eslovênia)
    - N=159 (pequeno para deep learning)
    - Pré-termo incluído = qualquer causa
    - Pesos do modelo não publicados
    - Validação externa necessária
    """)

    st.markdown("---")
    st.caption("App educacional desenvolvido para o CBGO · Dr. Alan Hatanaka")
