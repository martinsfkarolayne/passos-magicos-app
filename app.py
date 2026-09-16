"""
App Streamlit - Fase 4 do Datathon (Case Passos Mágicos)

Carrega o modelo treinado e estima a probabilidade de um aluno entrar em
risco de defasagem, a partir dos indicadores informados.

Modelo em uso: provisório (modelo_provisorio.pkl), até a conclusão da Fase 3.
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Passos Mágicos - Risco de Defasagem",
    page_icon="✨",
    layout="centered",
)

st.title("✨ Passos Mágicos")
st.subheader("Previsão de risco de defasagem escolar")

st.write(
    "Preencha os indicadores do aluno abaixo e clique em **Calcular risco** "
    "para estimar a probabilidade dele entrar em risco de defasagem."
)

st.caption("Versão com modelo provisório — será atualizada com o modelo final da equipe.")

@st.cache_resource
def carregar_modelo():
    dados_modelo = joblib.load("modelo_provisorio.pkl")
    return dados_modelo["modelo"], dados_modelo["features"]

modelo, features = carregar_modelo()

st.markdown("### Indicadores do aluno")
st.caption("Escala de 0 (mais baixo) a 10 (mais alto).")

col1, col2 = st.columns(2)

with col1:
    ida = st.slider("IDA - Desempenho acadêmico", 0.0, 10.0, 6.5, 0.1)
    ieg = st.slider("IEG - Engajamento", 0.0, 10.0, 7.5, 0.1)
    iaa = st.slider("IAA - Autoavaliação", 0.0, 10.0, 7.5, 0.1)
    ips = st.slider("IPS - Psicossocial", 0.0, 10.0, 6.5, 0.1)
    ipp = st.slider("IPP - Psicopedagógico", 0.0, 10.0, 7.0, 0.1)

with col2:
    ipv = st.slider("IPV - Ponto de virada", 0.0, 10.0, 7.5, 0.1)
    inde = st.slider("INDE - Nota global do aluno", 0.0, 10.0, 7.0, 0.1)
    mat = st.slider("Nota de Matemática", 0.0, 10.0, 6.0, 0.1)
    por = st.slider("Nota de Português", 0.0, 10.0, 6.5, 0.1)

if st.button("Calcular risco", type="primary"):
    entrada = pd.DataFrame(
        [[ida, ieg, iaa, ips, ipp, ipv, inde, mat, por]],
        columns=features,
    )

    probabilidade = modelo.predict_proba(entrada)[0][1]
    percentual = probabilidade * 100

    st.markdown("### Resultado")
    st.metric("Probabilidade de risco de defasagem", f"{percentual:.1f}%")

    if percentual >= 60:
        st.error("Risco alto — recomenda-se atenção prioritária e acompanhamento próximo.")
    elif percentual >= 30:
        st.warning("Risco moderado — vale monitorar a evolução do aluno nos próximos ciclos.")
    else:
        st.success("Risco baixo — indicadores dentro do esperado no momento.")

st.divider()
st.caption("Datathon Fase 5 · Pós-graduação FIAP · Case Passos Mágicos")
