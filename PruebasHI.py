import scipy.io
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import funciones as mifu
mifu.importar_funciones()

sujeto = 1
prediccion = 'estimulo'

datos = mifu.extraer(sujeto)
datos_ordenados = mifu.ordenar(datos)
datos_normalizados = mifu.normalizar(datos_ordenados, 'prom0')
datos_divididos = mifu.dividir_datos(datos_normalizados, 0.25, prediccion)
[[x_entrenamiento, e_entrenamiento], [x_prueba, e_prueba]] = datos_divididos

# mifu.graficar(datos, 1)
# plt.plot(np.concatenate(datos_normalizados[0][:,:,0]))
# plt.show()
# plt.plot(datos_normalizados)

###############################################################################################
# Perceptron simple:
###############################################################################################

if 1 == 0:
    # El perceptron simple no puede manejar estructuras de datos como 6 bloques de 4096 datos, uno en cada neurona de entrada.
    # Para poder utilizarlo hace falta aplanar los 6 canales, en un solo vector de 6*4096 datos, y mandar cada valor a una neurona de entrada.
    # Habra tambien, una neurona de salida por cada etiqueta.

    x_entrenamiento_aplanado = x_entrenamiento.reshape(x_entrenamiento.shape[0], x_entrenamiento.shape[1] * x_entrenamiento.shape[2])
    x_prueba_aplanado = x_prueba.reshape(x_prueba.shape[0], x_prueba.shape[1] * x_prueba.shape[2]) # aplanado de bloques

                            # (cantidad de capas ocultas, epocas, semilla, mostrar progreso en epocas, tolerancia de criterio de parada)
    mod_perceptron = MLPClassifier(hidden_layer_sizes = (), max_iter = 1000, random_state = 0, verbose = False, tol =1e-4)
    mod_perceptron.fit(x_entrenamiento_aplanado, e_entrenamiento)

    e_pred = mod_perceptron.predict(x_prueba_aplanado)
    accuracy = accuracy_score(e_prueba, e_pred)
    accuracy

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

    # Voy a barrer la fraccion de entrenamiento para ver las distintas performances:
    if 1 == 0:
    p = 0.15
    accuracy = []
    p_plot = []
    while p < 0.3:
        datos_divididos = mifu.dividir_datos(datos_normalizados, p, prediccion)
        [[x_entrenamiento, e_entrenamiento], [x_prueba, e_prueba]] = datos_divididos
        x_entrenamiento_aplanado = x_entrenamiento.reshape(x_entrenamiento.shape[0], x_entrenamiento.shape[1] * x_entrenamiento.shape[2])
        x_prueba_aplanado = x_prueba.reshape(x_prueba.shape[0], x_prueba.shape[1] * x_prueba.shape[2]) # aplanado de bloques
        mod_perceptron = MLPClassifier(hidden_layer_sizes = (), max_iter = 1000, random_state = 0, verbose = False, tol =1e-4)
        mod_perceptron.fit(x_entrenamiento_aplanado, e_entrenamiento)
        e_pred = mod_perceptron.predict(x_prueba_aplanado)
        accuracy.append(accuracy_score(e_prueba, e_pred))
        p_plot.append(p)
        p += 0.005
    plt.plot(p_plot, accuracy)
    plt.show()

###############################################################################################
# Perceptron multicapa:
###############################################################################################
if 1 == 1:

    x_entrenamiento_aplanado = x_entrenamiento.reshape(
        x_entrenamiento.shape[0], x_entrenamiento.shape[1] * x_entrenamiento.shape[2])
    x_prueba_aplanado = x_prueba.reshape(
        x_prueba.shape[0], x_prueba.shape[1] * x_prueba.shape[2])
    
    neuronas_entrada = x_entrenamiento_aplanado.shape[1]
    ocultas = []
    if prediccion == 'modalidad' or 'artefacto':
        neuronas_salida = 2
    elif prediccion == 'estimulo':
        neuronas_salida = 11

    modelo = mifu.MLP(neuronas_entrada, ocultas, neuronas_salida)
    mifu.MLP.entrenar(modelo, 1000, x_entrenamiento_aplanado, e_entrenamiento)

    epocas = 1000
    datos = torch.from_numpy(x_entrenamiento_aplanado).float()
    etiquetas = torch.from_numpy(e_entrenamiento).long()
    print(etiquetas.shape)
    print(datos.shape)
    optimizador = optim.Adam(modelo.parameters(), lr = 0.001)
    criterio = nn.CrossEntropyLoss()

    for epoch in range(epocas):
        modelo.train()
        optimizador.zero_grad()
        outputs = modelo(datos)
        loss = criterio(outputs, etiquetas) # aca esta el error
        loss.backward()
        optimizador.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epocas}, Loss: {loss.item()}')