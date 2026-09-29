"""
App Streamlit - Datathon (Case Passos Mágicos)

Formulário e lógica de predição conforme o guia de integração da Thaty
("Orientações para Streamlit" - Fase 3, modelo preditivo). Carrega o
modelo treinado (XGBoost) e estima a probabilidade de um aluno entrar em
risco no ciclo atual (Em_Risco_Vigente), a partir dos indicadores
informados.
"""

import streamlit as st
import pandas as pd
import joblib

# 1. Carrega modelo e colunas salvas
dados_app = joblib.load("modelo_xgb_passos_magicos.pkl")
modelo = dados_app["modelo_xgb"]
colunas_treino = dados_app["colunas_treino"]

st.title("🛡️ Diagnóstico Preditivo de Risco — Passos Mágicos")

st.write(
    "Preencha os indicadores do aluno abaixo e clique em **Calcular Risco** "
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

# 2. Formulário
col1, col2 = st.columns(2)

with col1:
    ida = st.number_input("IDA (Desempenho)", 0.0, 10.0, 6.5)
    ieg = st.number_input("IEG (Engajamento)", 0.0, 10.0, 7.0)
    iaa = st.number_input("IAA (Autoavaliação)", 0.0, 10.0, 8.0)
    ips = st.number_input("IPS (Psicossocial)", 0.0, 10.0, 6.0)

with col2:
    ipp = st.number_input("IPP (Psicopedagógico)", 0.0, 10.0, 7.0)
    ipv = st.number_input("IPV (Ponto de Virada)", 0.0, 10.0, 7.5)
    # Se o usuário não ajustar Matemática/Português, o valor acompanha o IDA
    # (regra definida pela equipe para o caso de nota não informada).
    mat = st.number_input("Nota de Matemática", 0.0, 10.0, ida)
    por = st.number_input("Nota de Português", 0.0, 10.0, ida)

genero = st.selectbox("Gênero", ["Feminino", "Masculino"])
# Categorias brutas exatamente como aparecem na base de treino (mesmas 10
# variações usadas para gerar as colunas one-hot do modelo). "Concluiu o 3º
# EM" é a categoria-base do encoding (drop_first=True) e por isso não tem
# coluna própria — selecioná-la deixa todas as colunas de instituição em 0,
# o que é o comportamento correto. As duas variações de "Programa de
# Apadrinhamento" (com A/a maiúscula/minúscula) existem porque a base bruta
# tem essa inconsistência de digitação; o modelo aprendeu as duas como
# colunas separadas, então mantemos ambas aqui.
instituicao = st.selectbox(
    "Instituição de Ensino",
    [
        "Escola Pública",
        "Rede Decisão",
        "Privada",
        "Privada - Programa de Apadrinhamento",
        "Privada - Programa de apadrinhamento",
        "Privada *Parcerias com Bolsa 100%",
        "Pública",
        "Escola JP II",
        "Nenhuma das opções acima",
        "Concluiu o 3º EM",
    ],
)
fase = st.selectbox("Fase", ["0", "1", "2", "3", "4", "5", "6", "7", "ALFA"])

# 3. Predição
if st.button("Calcular Risco"):
    dados_input = {
        "IDA": ida,
        "IEG": ieg,
        "IAA": iaa,
        "IPS": ips,
        "IPP": min(max(ipp, 0.0), 10.0),
        "IPV": ipv,
        "Mat": mat,
        "Por": por,
        "Gênero": genero,
        "Instituição de ensino": instituicao,
        "Fase": fase,
    }

    df_raw = pd.DataFrame([dados_input])
    df_encoded = pd.get_dummies(df_raw).astype(int)

    # Garante estrutura idêntica de colunas do treino
    df_final = pd.DataFrame(0, index=[0], columns=colunas_treino)
    for col in df_encoded.columns:
        if col in df_final.columns:
            df_final[col] = df_encoded[col]
    # .astype(int) trunca os indicadores numéricos (6.5 viraria 6); repõe os
    # valores originais com casas decimais depois do alinhamento de colunas.
    for num_col in ["IDA", "IEG", "IAA", "IPS", "IPP", "IPV", "Mat", "Por"]:
        df_final[num_col] = dados_input[num_col]

    prob_risco = modelo.predict_proba(df_final)[0][1]

    st.subheader(f"Probabilidade de Risco: {prob_risco * 100:.1f}%")
    if prob_risco >= 0.70:
        st.error("🚨 Alto Risco: Intervenção psicopedagógica urgente recomendada.")
    elif prob_risco >= 0.40:
        st.warning("⚠️ Risco Moderado: Monitorar engajamento e assiduidade.")
    else:
        st.success("✅ Aluno Seguro: Desempenho dentro da média esperada.")

st.divider()
st.caption("Datathon Fase 5 · Pós-graduação FIAP · Case Passos Mágicos")
