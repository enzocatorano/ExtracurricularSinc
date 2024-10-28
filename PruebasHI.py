import scipy.io
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import ExtracurricularSinc.funciones as mifu
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
# Perceptron multicapa:
###############################################################################################
if 1 == 1:

    x_entrenamiento_aplanado = x_entrenamiento.reshape(
        x_entrenamiento.shape[0], x_entrenamiento.shape[1] * x_entrenamiento.shape[2])
    x_prueba_aplanado = x_prueba.reshape(
        x_prueba.shape[0], x_prueba.shape[1] * x_prueba.shape[2])
    
    neuronas_entrada = x_entrenamiento_aplanado.shape[1]
    ocultas = [33,33]

    if prediccion == 'modalidad' or prediccion == 'artefacto':
        neuronas_salida = 2
    elif prediccion == 'estimulo':
        neuronas_salida = 11

    modelo = mifu.MLP(neuronas_entrada, ocultas, neuronas_salida)
    mifu.MLP.entrenar(modelo, 5000, x_entrenamiento_aplanado, e_entrenamiento, 1)

    precision, predicciones = mifu.MLP.evaluar(modelo, x_prueba_aplanado, e_prueba)