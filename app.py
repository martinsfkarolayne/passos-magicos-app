"""
App Streamlit - Fase 4 do Datathon (Case Passos Mágicos)

O que este aplicativo faz:
- Carrega um modelo já treinado (arquivo .pkl)
- Deixa a pessoa digitar os indicadores de um aluno
- Mostra a probabilidade desse aluno estar em risco de defasagem

IMPORTANTE: por enquanto o modelo carregado aqui é PROVISÓRIO
(modelo_provisorio.pkl), feito só para testar o aplicativo. Quando a
Thaty terminar o modelo real da Fase 3, é só substituir o arquivo do
modelo (e ajustar a lista de campos, se as colunas forem diferentes).
"""

import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------------
# 1. Configuração da página
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Passos Mágicos - Risco de Defasagem",
    page_icon="✨",
    layout="centered",
)

st.title("✨ Passos Mágicos")
st.subheader("Previsão de risco de defasagem escolar")

st.write(
    "Preencha os indicadores do aluno abaixo e clique em **Calcular risco** "
    "para ver a probabilidade dele entrar em risco de defasagem."
)

st.info(
    "⚠️ Este app está usando um **modelo provisório**, criado apenas para "
    "testar o funcionamento do aplicativo. Assim que o modelo definitivo "
    "da equipe estiver pronto, ele será atualizado.",
    icon="⚠️",
)

# ---------------------------------------------------------------
# 2. Carregar o modelo treinado
# ---------------------------------------------------------------
@st.cache_resource
def carregar_modelo():
    dados_modelo = joblib.load("modelo_provisorio.pkl")
    return dados_modelo["modelo"], dados_modelo["features"]

modelo, features = carregar_modelo()

# ---------------------------------------------------------------
# 3. Formulário de entrada com os indicadores do aluno
# ---------------------------------------------------------------
st.markdown("### Indicadores do aluno")
st.caption("Todos os indicadores vão de 0 (mais baixo) a 10 (mais alto).")

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

# ---------------------------------------------------------------
# 4. Botão de calcular e exibição do resultado
# ---------------------------------------------------------------
if st.button("Calcular risco", type="primary"):
    entrada = pd.DataFrame(
        [[ida, ieg, iaa, ips, ipp, ipv, inde, mat, por]],
        columns=features,
    )

    probabilidade = modelo.predict_proba(entrada)[0][1]  # probabilidade da classe "em risco"
    percentual = probabilidade * 100

    st.markdown("### Resultado")
    st.metric("Probabilidade de risco de defasagem", f"{percentual:.1f}%")

    if percentual >= 60:
        st.error(
            "🔴 Risco **alto**. Recomenda-se atenção prioritária e "
            "acompanhamento próximo desse aluno."
        )
    elif percentual >= 30:
        st.warning(
            "🟡 Risco **moderado**. Vale a pena monitorar a evolução "
            "desse aluno nos próximos ciclos."
        )
    else:
        st.success(
            "🟢 Risco **baixo**. O aluno está com indicadores dentro do "
            "esperado no momento."
        )

st.divider()
st.caption(
    "Datathon Fase 5 · Pós-graduação FIAP · Case Passos Mágicos"
)
