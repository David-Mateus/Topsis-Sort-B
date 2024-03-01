import numpy as np

def vikor(X, P, D, w):
    """
    Função para implementar o método VIKOR com correções para divisão por zero e valores inválidos.

    Argumentos:
        X: Matriz de decisão (m x n).
        P: Matriz de perfis dominantes (p x n).
        D: Matriz de domínio (2 x n).
        w: Vetor de pesos (n x 1).

    Retorna:
        Matriz de classificação (m x 2).
    """

    # Normalização da matriz de decisão
    R = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            R[i, j] = X[i, j] / D[0, j]

    # Ponderação da matriz de decisão normalizada
    V = np.zeros_like(R)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            V[i, j] = w[j] * R[i, j]

    # Cálculo das soluções ideal e anti-ideal
    V_ideal = np.max(V, axis=0)
    V_anti_ideal = np.min(V, axis=0)

    # Cálculo das distâncias euclidianas
    d_plus = np.zeros(X.shape[0])
    d_minus = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        d_plus[i] = np.sqrt(np.sum((V[i, :] - V_ideal)**2))
        d_minus[i] = np.sqrt(np.sum((V[i, :] - V_anti_ideal)**2))

    # Corrigindo divisão por zero
    d_plus[d_plus == 0] = np.finfo(float).eps
    d_minus[d_minus == 0] = np.finfo(float).eps

    # Cálculo do coeficiente de aproximação
    Cl = d_minus / (d_plus + d_minus)

    # Cálculo das distâncias dos perfis dominantes
    d_plus_p = np.zeros(P.shape[0])
    d_minus_p = np.zeros(P.shape[0])
    for k in range(P.shape[0]):
        d_plus_p[k] = np.sqrt(np.sum((P[k, :] - V_ideal)**2))
        d_minus_p[k] = np.sqrt(np.sum((P[k, :] - V_anti_ideal)**2))

    # Corrigindo divisão por zero
    d_plus_p[d_plus_p == 0] = np.finfo(float).eps
    d_minus_p[d_minus_p == 0] = np.finfo(float).eps

    # Cálculo do coeficiente de aproximação dos perfis dominantes
    Cl_p = d_minus_p / (d_plus_p + d_minus_p)

    # Classificação das alternativas e perfis
    C = np.zeros((X.shape[0], 2))
    for i in range(X.shape[0]):
        if Cl[i] >= Cl_p[0]:
            C[i, 0] = 1
        else:
            for k in range(1, P.shape[0]):
                if Cl_p[k-1] > Cl[i] >= Cl_p[k]:
                    C[i, 0] = k + 1
                    break
        C[i, 1] = Cl[i]

    return C

# Exemplo de uso
X = np.array([[5, 8, 7], [7, 6, 9], [9, 4, 8], [8, 7, 9]])
P = np.array([[3, 9, 10], [10, 5, 8]])
D = np.array([[3, 10, 10], [1, 5, 8]])
w = np.array([0.4, 0.3, 0.3])


C = vikor(X, P, D, w)

print(C)
# Encontrando a melhor solução
best_solution_index = np.argmax(C[:, 1])  # Índice da linha com o maior coeficiente de aproximação
best_solution = X[best_solution_index]  # Melhor solução
best_profile = int(C[best_solution_index, 0])  # Perfil dominante da melhor solução

print("Melhor solução:", best_solution)
print("Perfil dominante da melhor solução:", best_profile)