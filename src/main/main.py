from topsis_sort_b import topsis_b_sort_profile_classification
import pandas as pd
import streamlit as st
import numpy as np
import re

def main():
    st.title("Topsis-Sort-B")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"]) # TODO: Validar

    st.write('Perfil dominante')
    # [[3, 1, 5]]
    container = st.container()

    # Cria as colunas
    col1, col2, col3 = container.columns(3)

    # Cria os inputs dentro das colunas
    num1 = col1.number_input("Primeiro número", min_value=1, max_value=10, step=1, key="num1")
    num2 = col2.number_input("Segundo número", min_value=1, max_value=10, step=1, key="num2")
    num3 = col3.number_input("Terceiro número", min_value=1, max_value=10, step=1, key="num3")

    values = st.slider(
        'Matriz de Domínio',
        1, 100, (25, 75))

    st.write('Pesos')
    weight_1 = st.slider(' ', 0.1, 1.0, 0.1)
    weight_2 = st.slider('  ', 0.1, 1.0, 0.1)
    weight_3 = st.slider('   ', 0.1, 1.0, 0.6)

    dominant_profiles = np.array([[num1, num2, num3]])
    domain_matrix = np.array([[values[0], values[0], values[0]], [values[1], values[1], values[1]]])
    weights = np.array([weight_1, weight_2, weight_3])

    if uploaded_file is not None:
        # Read the content of the uploaded file
        data = np.loadtxt(uploaded_file, delimiter=',', skiprows=1)
# Matriz de decisão (excluindo a última coluna que representa as vendas)
        decision_matrix = data[:, :-1]

        # Run the TOPSIS analysis
        classification_result, best_solution, best_profile = topsis_b_sort_profile_classification(decision_matrix, domain_matrix, dominant_profiles, weights)

        # Display the results
        st.write("Classification Result:")
        st.table(pd.DataFrame(classification_result, columns=["Dominant Profile", "Approximation Coefficient"]))

        st.write("Best Solution:")
        st.table(pd.DataFrame(best_solution, columns=[f"Feature: " ]))

        st.write("Dominant Profile of the Best Solution:", best_profile)

    else:
        st.warning("Please upload a CSV file.")


if __name__ == "__main__":
    main()

# def main():
#     st.title("Topsis-Sort-B")

#     # Fazer upload do arquivo CSV
#     uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
#     dominant_profiles = st.text_area("Dominant Profiles (comma-separated values): ")
#     domain_matrix = st.text_area("Domain Matrix (comma-separated values, one row per line): ")
#     weights = st.text_area("Weights (comma-separated values): ")

#     if st.button("Run Analysis"):
#         try:
#             if uploaded_file:
#                 decision_matrix = pd.read_csv(uploaded_file, header=None).values
#             else:
#                 decision_matrix = np.array([list(map(int, row.split(','))) for row in uploaded_file.split('\n')])
            
#             dominant_profiles = np.array([list(map(int, re.split(r'[,\s]+', dominant_profiles.strip())))])
#             domain_matrix = np.array([list(map(int, row.split(','))) for row in domain_matrix.split('\n')])
#             weights = np.array(list(map(float, weights.split(','))))

#             # Executa a análise TOPSIS
#             classification_result, best_solution, best_profile = topsis_b_sort_profile_classification(decision_matrix, domain_matrix, dominant_profiles, weights)

#             # Exibe os resultados
#             st.write("Classification Result:")
#             st.table(pd.DataFrame(classification_result, columns=["Dominant Profile", "Approximation Coefficient"]))

#             st.write("Best Solution:")
#             st.table(pd.DataFrame(best_solution, columns=[f"Feature {i}" for i in range(1, len(best_solution)+1)]))

#             st.write("Dominant Profile of the Best Solution:", best_profile)
        
#         except Exception as e:
#             st.error(f"An error occurred: {str(e)}")

# if __name__ == "__main__":
#     main()
