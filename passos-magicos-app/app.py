"""
App Streamlit - Datathon (Case Passos Mágicos)

Carrega o modelo treinado (XGBoost) e estima a probabilidade de um aluno
entrar em risco no ciclo atual (Em_Risco_Vigente), a partir dos
indicadores informados.
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
st.subheader("Diagnóstico preditivo de risco (ciclo atual)")

st.write(
    "Preencha os indicadores do aluno abaixo e clique em **Calcular risco** "
    "para estimar a probabilidade dele entrar em risco no ciclo atual."
)

with st.expander("O que significa cada indicador?"):
    st.markdown(
        "- **IDA**: desempenho do aluno nas avaliações acadêmicas realizadas pela Passos Mágicos.\n"
        "- **IEG**: nível de participação e envolvimento do aluno nas atividades propostas.\n"
        "- **IAA**: como o próprio aluno percebe seu desempenho e evolução.\n"
        "- **IPS**: aspectos emocionais e sociais que podem impactar o aprendizado.\n"
        "- **IPP**: resultado das avaliações psicopedagógicas feitas com o aluno.\n"
        "- **IPV**: o quanto o aluno já avançou rumo à transformação que o programa busca.\n"
        "- **Mat / Por**: notas do aluno nas avaliações de Matemática e Português.\n"
        "- **Fase**: fase atual do aluno dentro do programa (0 a 7, ou ALFA).\n\n"
        "O modelo não usa o indicador IAN, pois ele é calculado a partir da "
        "própria defasagem escolar, o que enviesaria a previsão."
    )

@st.cache_resource
def carregar_modelo():
    dados = joblib.load("modelo_xgb_passos_magicos.pkl")
    return dados["modelo_xgb"], dados["colunas_treino"]

modelo, colunas_treino = carregar_modelo()

st.markdown("### Indicadores do aluno")
st.caption("Indicadores numéricos em escala de 0 a 10.")

col1, col2 = st.columns(2)

with col1:
    ida = st.slider("IDA - Desempenho acadêmico", 0.0, 10.0, 6.5, 0.1)
    ieg = st.slider("IEG - Engajamento", 0.0, 10.0, 7.0, 0.1)
    iaa = st.slider("IAA - Autoavaliação", 0.0, 10.0, 8.0, 0.1)
    ips = st.slider("IPS - Psicossocial", 0.0, 10.0, 6.0, 0.1)

with col2:
    ipp = st.slider("IPP - Psicopedagógico", 0.0, 10.0, 7.0, 0.1)
    ipv = st.slider("IPV - Ponto de virada", 0.0, 10.0, 7.5, 0.1)
    # Se o usuário não ajustar Matemática/Português, o valor acompanha o IDA
    # (regra definida pela equipe para o caso de nota não informada).
    mat = st.slider("Nota de Matemática", 0.0, 10.0, ida, 0.1)
    por = st.slider("Nota de Português", 0.0, 10.0, ida, 0.1)

genero = st.selectbox("Gênero", ["Feminino", "Masculino"])
instituicao = st.selectbox(
    "Instituição de ensino", ["Escola Pública", "Rede Decisão", "Privada", "Outra"]
)
fase = st.selectbox("Fase", ["0", "1", "2", "3", "4", "5", "6", "7", "ALFA"])

if st.button("Calcular risco", type="primary"):
    entrada = {
        "IDA": ida,
        "IEG": ieg,
        "IAA": iaa,
        "IPS": ips,
        "IPP": min(max(ipp, 0.0), 10.0),  # trava de segurança
        "IPV": ipv,
        "Mat": mat,
        "Por": por,
        "Gênero": genero,
        "Instituição de ensino": instituicao,
        "Fase": fase,
    }

    df_raw = pd.DataFrame([entrada])
    df_encoded = pd.get_dummies(df_raw).astype(int)

    # Garante a mesma estrutura de colunas usada no treino
    df_final = pd.DataFrame(0, index=[0], columns=colunas_treino)
    for col in df_encoded.columns:
        if col in df_final.columns:
            df_final[col] = df_encoded[col]
    # .astype(int) trunca os indicadores numéricos (6.5 viraria 6); repõe os
    # valores originais com casas decimais depois do alinhamento de colunas.
    for num_col in ["IDA", "IEG", "IAA", "IPS", "IPP", "IPV", "Mat", "Por"]:
        df_final[num_col] = entrada[num_col]

    probabilidade = modelo.predict_proba(df_final)[0][1]
    percentual = probabilidade * 100

    st.markdown("### Resultado")
    st.metric("Probabilidade de risco", f"{percentual:.1f}%")

    if percentual >= 70:
        st.error("🚨 Alto risco - encaminhar para suporte psicopedagógico e reforço.")
    elif percentual >= 40:
        st.warning("⚠️ Risco moderado - alerta preventivo de engajamento e assiduidade.")
    else:
        st.success("✅ Sem risco - aluno no fluxo regular de aprendizado.")

st.divider()
st.caption("Datathon Fase 5 · Pós-graduação FIAP · Case Passos Mágicos")
