# Ignora a primeira linha
with open('./advertising.csv', 'r') as arquivo:
    linhas = arquivo.readlines()
dados = linhas[1:]

for linha in dados:
    valores = linha.strip().split(',')
    # Faça o que precisar com os valores
    print(valores)