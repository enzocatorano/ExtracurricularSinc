import funciones as mifu
import scipy.io
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import random
mifu.importar_funciones()

sujeto = 1
prediccion = 'estimulo'

datos = mifu.extraer(sujeto)
datos_ordenados = mifu.ordenar(datos)
datos_normalizados = mifu.normalizar(datos_ordenados)
datos_divididos = mifu.dividir_datos(datos_normalizados, 0.3, prediccion)

mifu.graficar(datos, 1)

[[x_entrenamiento, e_entrenamiento], [x_prueba, e_prueba]] = datos_divididos

# Perceptron simple:
    # El perceptron simple no puede manejar estructuras de datos como 6 bloques de 4096 datos, uno en cada neurona de entrada.
    # Para poder utilizarlo hace falta aplanar los 6 canales, en un solo vector de 6*4096 datos, y mandar cada valor a una neurona de entrada.
    # Habra tambien, una neurona de salida por cada etiqueta.

x_entrenamiento_aplanado = x_entrenamiento.reshape(x_entrenamiento.shape[0], x_entrenamiento.shape[1] * x_entrenamiento.shape[2])
x_prueba_aplanado = x_prueba.reshape(x_prueba.shape[0], x_prueba.shape[1] * x_prueba.shape[2]) # aplanado de bloques

                        # (cantidad de capas ocultas, epocas, semilla, mostrar progreso en epocas, tolerancia de criterio de parada)
mod_perceptron = MLPClassifier(hidden_layer_sizes = (), max_iter = 1000, random_state = 0, verbose = False, tol =1e-4)
mod_perceptron.fit(x_entrenamiento_aplanado, e_entrenamiento)

# Graficar
if 1 == 1:
    perdida = mod_perceptron.loss_curve_ # progreso por epocas
    plt.plot(perdida, label = "Pérdida")
    plt.xlabel("Épocas", fontsize=12, color = 'white')
    plt.ylabel("Pérdida", fontsize=12, color = 'white')
    plt.title("Evolución de la performance del perceptrón simple", fontsize=24, color = 'white')
    ax = plt.gca()
    ax.set_facecolor('black')
    plt.gcf().set_facecolor('black')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, facecolor='black', edgecolor='white', labelcolor='white')
    plt.tight_layout()
    plt.show()

e_pred = mod_perceptron.predict(x_prueba_aplanado)
accuracy = accuracy_score(e_prueba, e_pred)
accuracy
