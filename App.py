import streamlit as st
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Analisador de Quebra de Serviço - Tênis")

st.markdown("""
Cole abaixo os dados da partida no formato CSV com as colunas:  
`game_id,server,receiver,server_points,receiver_points,server_aces,server_double_faults,server_1st_serve_pct,game_winner,break_occurred`  
(Exemplo no final)
""")

data_input = st.text_area("Cole os dados aqui (CSV)", height=200)

if st.button("Analisar dados"):

    if not data_input.strip():
        st.error("Por favor, cole os dados antes de analisar!")
    else:
        try:
            data = pd.read_csv(StringIO(data_input))
            st.success("Dados carregados com sucesso!")
            st.write("Prévia dos dados:")
            st.dataframe(data.head())

            # Selecionar features e alvo
            features = ['server_points', 'receiver_points', 'server_aces', 'server_double_faults', 'server_1st_serve_pct']
            X = data[features]
            y = data['break_occurred']

            # Dividir treino/teste
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Modelo simples de regressão logística
            model = LogisticRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            st.write(f"**Acurácia do modelo:** {acc:.2f}")

            st.write("Relatório de classificação:")
            st.text(classification_report(y_test, y_pred))

            # Importância das features
            coef = pd.Series(model.coef_[0], index=features)
            st.write("Importância das variáveis:")
            st.bar_chart(coef)

            # Gráfico de correlações entre features
            st.write("Mapa de calor das correlações:")
            plt.figure(figsize=(8,6))
            sns.heatmap(data[features + ['break_occurred']].corr(), annot=True, cmap='coolwarm')
            st.pyplot(plt)

        except Exception as e:
            st.error(f"Erro ao processar os dados: {e}")

st.markdown("""
---

### Exemplo de dados que você pode colar:
