# Exemplo de Machine Learning para prever preço da pizza

import os 
os.system('cls') # limpar a tela

# Importando Matplotlib e Numpy
import matplotlib.pyplot as plt
import matplotlib as mat
import numpy as np

# Importando o módulo de Regressão Linear do scikit-learn
import sklearn
from sklearn.linear_model import LinearRegression

# Diâmetros (cm)
Diametros = [[7], [10], [15], [30], [45]]

# Preços (R$)
Precos = [[8], [11], [16], [38.5], [52]]

plt.figure()
plt.xlabel('Diâmetro(cm)')
plt.ylabel('Preço(R$)')
plt.title('Diâmetro x Preço')
plt.plot(Diametros, Precos, 'k.')
plt.axis([0, 60, 0, 60])
plt.grid(True)
plt.show()

# Preparando os dados de treino

# Vamos chamar de X os dados de diâmetro da Pizza.
X = Diametros ##[[7], [10], [15], [30], [45]]

# Vamos chamar de Y os dados de preço da Pizza.
Y = Precos ##[[8], [11], [16], [38.5], [52]]

# Criando o modelo
modelo = LinearRegression()

# Treinando o modelo
modelo.fit(X, Y)

# Prevendo o preço de uma pizza de 20 cm de diâmetro
tamanhopizza = int(input('\nDigite o tamanho da pizza: '))
print("Uma pizza com esse diâmetro deve custar: R$%.2f" % modelo.predict([[tamanhopizza]]))

tec = input("pressione [ENTER] para continuar...")

# Coeficientes
print('Coeficiente: \n', modelo.coef_)

# MSE (mean square error)
print("MSE: %.2f" % np.mean((modelo.predict(X) - Y) ** 2))

# Score de variação: 1 representa predição perfeita
print('Score de variação: %.2f' % modelo.score(X, Y))

tec = input("pressione [ENTER] para continuar...")

# Scatter Plot representando a regressão linear
plt.scatter(X, Y,  color = 'black')
plt.plot(X, modelo.predict(X), color = 'blue', linewidth = 3)
plt.xlabel('X')
plt.ylabel('Y')
plt.xticks(())
plt.yticks(())

plt.show()

tec = input("pressione [ENTER] para finalizar...")
