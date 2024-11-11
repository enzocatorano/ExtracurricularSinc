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
sujeto = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
prediccion = ['modalidad', 'estimulo', 'artefacto']
normalizacion = ['minmax', 'prom0']
ocultas = [[], [256], [256, 256], [256, 256, 256], [256, 256, 256, 256, 256, 256, 256]]
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
                    np.bincount(predicciones)
                    valores.append(precision)
                error = np.std(valores, ddof=1)/divisor
                valor = np.mean(valores)
                carac = 'S' + str(i) + '/' + k + '/' + j + '/' + str([neuronas_entrada] + l + [neuronas_salida]).replace(',','')
                mifu.registrar(valor, error, carac, fecha, hito)
    print('Listo el sujeto ' + str(i) + '.')


ruta = os.path.join('ExtracurricularSinc', 'historial.csv')
with open(ruta, mode='r', newline='', encoding='latin-1') as archivo:
    lector_csv = csv.reader(archivo)
    datos = []
    for i in lector_csv:
        datos.append(i)

datos_arreglados = []
for i in datos:
    j = []
    j.append(float(i[0]))
    j.append(float(i[1]))
    k = i[2].split('/')
    k[3] = k[3][1:-1].split(' ')
    m = []
    for n in k[3]:
        m.append(int(n))
    j.append(int(k[0][1:]))
    j.append(k[1])
    j.append(k[2])
    j.append(m)
    j.append(i[3])
    j.append(i[4])
    datos_arreglados.append(j)
datos = datos_arreglados
datos[0]

# Perceptron simple con minmax:
prom = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
error = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
for i in datos:
    if i[3] == 'minmax' and len(i[5]) == 2:
        prom[i[2] - 1].append(i[0])
        error[i[2] - 1].append(i[1])
prom
error
promg = np.array([0.0,0.0,0.0])
errorg = np.array([0.0,0.0,0.0])
for i in prom:
    promg[0] += i[0]
    promg[1] += i[1]
    promg[2] += i[2]
for i in error:
    errorg[0] += i[0]**2
    errorg[1] += i[1]**2
    errorg[2] += i[2]**2
errorg = (np.sqrt(errorg)) / len(prom)
promg = promg / len(prom)
promg, errorg

print(promg)
for i in prom:
    print(i)

# Perceptron simple con prom0:
prom = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
error = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
for i in datos:
    if i[3] == 'prom0' and len(i[5]) == 2:
        prom[i[2] - 1].append(i[0])
        error[i[2] - 1].append(i[1])
prom
error
promg = np.array([0.0,0.0,0.0])
errorg = np.array([0.0,0.0,0.0])
for i in prom:
    promg[0] += i[0]
    promg[1] += i[1]
    promg[2] += i[2]
for i in error:
    errorg[0] += i[0]**2
    errorg[1] += i[1]**2
    errorg[2] += i[2]**2
errorg = (np.sqrt(errorg)) / len(prom)
promg = promg / len(prom)
promg, errorg

print(promg)
for i in prom:
    print(i)

# MLP con ambas:
norm = ['minmax', 'prom0']
nco = [1, 2, 3, 7]
for b in norm:
    for a in nco:
        prom = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
        error = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
        for i in datos:
            if i[3] == b and len(i[5]) == a + 2:
                prom[i[2] - 1].append(i[0])
                error[i[2] - 1].append(i[1])
        # prom
        # error
        promg = np.array([0.0,0.0,0.0])
        errorg = np.array([0.0,0.0,0.0])
        for i in prom:
            promg[0] += i[0]
            promg[1] += i[1]
            promg[2] += i[2]
        for i in error:
            errorg[0] += i[0]**2
            errorg[1] += i[1]**2
            errorg[2] += i[2]**2
        errorg = (np.sqrt(errorg)) / len(prom)
        promg = promg / len(prom)
        promg.tolist()

# minmax vs prom0:
from scipy import stats
minmax = 0
prom0 = 0
nulo = 0
alpha = 0.05
i = 0
while i < len(datos)/15:
    for j in range(i*15, (i+1)*15):
        minmaxp = datos[j][0]
        minmaxe = datos[j][1]
        prom0p = datos[j+15][0]
        prom0e = datos[j+15][1]
        t = (minmaxp - prom0p)/(minmaxe**2 + prom0e**2)
        df = (minmaxe**4 + prom0e**4) / ((minmaxe**4 / 4) + (prom0e**4 / 4))
        p = stats.t.sf(np.abs(t), df) * 2  # Multiplicar por 2 para prueba bilateral
        if p > alpha:
            nulo += 1
        elif p < alpha and minmaxp > prom0p:
            minmax += 1
        elif p < alpha and minmaxp < prom0p:
            prom0 += 1
    i += 2
minmax, prom0, nulo

# Tamaño de los perceptrones:
norm = ['minmax', 'prom0']
nco = [0, 1, 2, 3, 7]
yminmax = []
eminmax = []
yprom0 = []
eprom0 = []
for b in norm:
    for a in nco:
        prom = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
        error = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
        for i in datos:
            if i[3] == b and len(i[5]) == a + 2:
                prom[i[2] - 1].append(i[0])
                error[i[2] - 1].append(i[1])
        # prom
        # error
        promg = np.array([0.0,0.0,0.0])
        errorg = np.array([0.0,0.0,0.0])
        for i in prom:
            promg[0] += i[0]
            promg[1] += i[1]
            promg[2] += i[2]
        for i in error:
            errorg[0] += i[0]**2
            errorg[1] += i[1]**2
            errorg[2] += i[2]**2
        errorg = (np.sqrt(errorg)) / len(prom)
        promg = promg / len(prom)
        if b == 'minmax':
            yminmax.append(promg)
            eminmax.append(errorg)
        elif b == 'prom0':
            yprom0.append(promg)
            eprom0.append(errorg)
yminmax = np.array(yminmax).T
eminmax = np.array(eminmax).T
yprom0 = np.array(yprom0).T
eprom0 = np.array(eprom0).T

x1 = []
x2 = []
for i in range(5):
    n1 = 0
    n2 = 0
    for j in range(len(datos[i][5]) - 1):
        n1 += datos[i][5][j+1]*datos[i][5][j] + datos[i][5][j+1]
        n2 += datos[i+5][5][j+1]*datos[i+5][5][j] + datos[i+5][5][j+1]
    x1.append(n1)
    x2.append(n2)
x = [x1, x2, x1]
x

for i in range(len(x)):
    plt.errorbar(x[i][1:], yminmax[i][1:], yerr = eminmax[i][1:], fmt='o', capsize=5)
plt.show()

# Probemos hacer una transformada de fourier discreta:
x = []
y = []
for i in range(10):
    x.append(random.random())
    y.append(random.random())
plt.scatter(x, y)
plt.show()

for k in range(len(x)):
    term = 0
    for n in range(len(x)):
        term += 1