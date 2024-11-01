import scipy.io
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import datetime
import csv
import ExtracurricularSinc.funciones as mifu

sujeto = 1
prediccion = 'estimulo'

datos = mifu.extraer(sujeto)
datos_ordenados = mifu.ordenar(datos)
datos_normalizados = mifu.normalizar(datos_ordenados, 'minmax')
datos_divididos = mifu.dividir_datos(datos_normalizados, 0.25, prediccion)
[[x_entrenamiento, e_entrenamiento], [x_prueba, e_prueba]] = datos_divididos

# mifu.graficar(datos, 1)
# plt.plot(np.concatenate(datos_normalizados[0][:,:,0]))
# plt.show()
# plt.plot(datos_normalizados)

###############################################################################################
# Perceptron simple/multicapa:
###############################################################################################
if 1 == 1:

    x_entrenamiento_aplanado = x_entrenamiento.transpose(0,2,1).reshape(
        x_entrenamiento.shape[0], x_entrenamiento.shape[1] * x_entrenamiento.shape[2])
    x_prueba_aplanado = x_prueba.transpose(0,2,1).reshape(
        x_prueba.shape[0], x_prueba.shape[1] * x_prueba.shape[2])
    
    neuronas_entrada = x_entrenamiento_aplanado.shape[1]
    ocultas = [256]

    if prediccion == 'modalidad' or prediccion == 'artefacto':
        neuronas_salida = 2
    elif prediccion == 'estimulo':
        neuronas_salida = 11

    modelo = mifu.MLP(neuronas_entrada, ocultas, neuronas_salida)
    mifu.MLP.entrenar(modelo, 1000, x_entrenamiento_aplanado, e_entrenamiento, 1, 1e-5, 0.0001)

    precision, predicciones = mifu.MLP.evaluar(modelo, x_prueba_aplanado, e_prueba)
    np.bincount(predicciones), np.bincount(e_prueba)

###############################################################################################
# Guardamos algunos de los valores de precision que fuimos obteniendo:
###############################################################################################
cantidad = 5
sujeto = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
prediccion = ['modalidad', 'estimulo', 'artefacto']
normalizacion = ['minmax', 'prom0']
ocultas = [[], [1024], [1024, 1024], [1024, 1024, 1024], [1024, 1024, 1024, 1024, 1024, 1024, 1024]]
fechas = ['10-10-2024', '14-10-2024', '15-10-2024']
divisor = math.sqrt(10)
for i in sujeto:
    datos = mifu.extraer(i)
    datos_ordenados = mifu.ordenar(datos)
    for k in normalizacion:
        datos_normalizados = mifu.normalizar(datos_ordenados, k)
        for j in prediccion:
            if j == 'modalidad' or prediccion == 'artefacto':
                neuronas_salida = 2
                frac = 0.4
                tol = 1e-5
                lr = 0.0001
            elif j == 'estimulo':
                neuronas_salida = 11
                frac = 0.25
                tol = 1e-5
                lr = 0.0001
            for l in ocultas:
                if k == 'minmax' and l == []:
                    fecha = fechas[0]
                    hito = 'Primer perceptron simple.'
                elif k == 'prom0' and l == []:
                    fecha = fechas[1]
                    hito = 'Cambié normalizacion mixmax a abs_prom0.'
                elif l != []:
                    fecha = fechas[2]
                    hito = 'Implmente MLP con ambas normalizaciones.'
                valores = []
                for n in range(cantidad):
                    print(n + 1)
                    datos_divididos = mifu.dividir_datos(datos_normalizados, frac, j)
                    [[x_entrenamiento, e_entrenamiento], [x_prueba, e_prueba]] = datos_divididos
                    x_entrenamiento_aplanado = x_entrenamiento.reshape(
                        x_entrenamiento.shape[0], x_entrenamiento.shape[1] * x_entrenamiento.shape[2])
                    x_prueba_aplanado = x_prueba.reshape(
                        x_prueba.shape[0], x_prueba.shape[1] * x_prueba.shape[2])
                    neuronas_entrada = x_entrenamiento_aplanado.shape[1]
                    modelo = mifu.MLP(neuronas_entrada, l, neuronas_salida)
                    mifu.MLP.entrenar(modelo, 1000, x_entrenamiento_aplanado, e_entrenamiento, 0, tol, lr)
                    precision, predicciones = mifu.MLP.evaluar(modelo, x_prueba_aplanado, e_prueba)
                    valores.append(precision)
                error = np.std(valores, ddof=1)/divisor
                valor = np.mean(valores)
                carac = 'S' + str(i) + '/' + k + '/' + j + '/' + str([neuronas_entrada] + l + [neuronas_salida]).replace(',','')
                mifu.registrar(valor, error, carac, fecha, hito)
    print('Listo el sujeto ' + i + '.')
