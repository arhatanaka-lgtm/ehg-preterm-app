# EHG + IA na Predição do Parto Pré-Termo

Dashboard educacional interativo baseado no artigo:

> **Goldsztejn U, Nehorai A.** Predicting preterm births from electrohysterogram recordings via deep learning. *PLoS ONE*. 2023;18(5):e0285219.  
> DOI: https://doi.org/10.1371/journal.pone.0285219 | PMID: 37167285

Desenvolvido para apresentação no **Congresso Brasileiro de Ginecologia e Obstetrícia (CBGO/FEBRASGO)** como parte da palestra *"O uso da inteligência artificial e genética na predição do parto pré-termo"*.

---

## Funcionalidades

| Aba | Conteúdo |
|-----|----------|
| 📊 Curvas ROC | Visualização interativa das curvas ROC dos 3 modelos (Clínico, EHG, Combinado) com pontos da Tabela 2 |
| 📈 Performance Detalhada | Sensibilidade, PPV e NPV a diferentes limiares de especificidade (50%, 70%, 90%) |
| 🔬 Análise de Bandas EHG | AUC por banda de frequência (B0–B3) e AUC vs. duração da gravação |
| 🌐 Comparação Global | Benchmark do método vs. fibronectina alfa, comprimento cervical e outros EHG |
| 👩‍⚕️ Risco Clínico (Aprox.) | Escore aproximado a partir dos Risk Ratios publicados das variáveis clínicas |
| 📚 Como Funciona | Explicação didática do pipeline: BPF → STFT → BiLSTM → Logistic Regression |

---

## Instalação local

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Aviso importante

> ⚠️ Esta ferramenta é exclusivamente **educacional**. Não é um dispositivo médico certificado e não substitui avaliação clínica individualizada. Os coeficientes β do modelo LASSO não foram publicados; o escore de risco clínico é uma **aproximação** baseada nos Risk Ratios brutos.

---

## Citação

```
Goldsztejn U, Nehorai A. Predicting preterm births from electrohysterogram recordings via 
deep learning. PLoS ONE. 2023;18(5):e0285219. doi: 10.1371/journal.pone.0285219.
```

Código original dos autores: https://github.com/uri-goldsztejn/Predicting_preterm_birth_from_EHG

---

## Desenvolvedor

Dr. Alan Hatanaka — Obstetra, Medicina Fetal e Alto Risco  
[@dralanhatanaka](https://instagram.com/dralanhatanaka)
