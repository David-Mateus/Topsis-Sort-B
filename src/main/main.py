from topsis_sort_b import topsis_b_sort_profile_classification
import pandas as pd
import streamlit as st
import numpy as np
import re

def main():
    st.title("Topsis-Sort-B")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    dominant_profiles = np.array([[3, 1, 5]])
    domain_matrix = np.array([[1, 1, 1], [100, 100, 100]])
    weights = np.array([0.2, 0.2, 0.6])

    if uploaded_file is not None:
        # Read the content of the uploaded file
        decision_matrix = pd.read_csv(uploaded_file, header=None).values

        # Run the TOPSIS analysis
        classification_result, best_solution, best_profile = topsis_b_sort_profile_classification(decision_matrix, domain_matrix, dominant_profiles, weights)

        # Display the results
        st.write("Classification Result:")
        st.table(pd.DataFrame(classification_result, columns=["Dominant Profile", "Approximation Coefficient"]))

        st.write("Best Solution:")
        st.table(pd.DataFrame(best_solution, columns=[f"Feature {i}" for i in range(1, len(best_solution)+1)]))

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
