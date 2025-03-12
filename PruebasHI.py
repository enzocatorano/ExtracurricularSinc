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
import funciones as mifu

sujeto = 1
prediccion = 'estimulo'

datos = mifu.extraer(sujeto)
datos_ordenados = mifu.ordenar(datos)
datos_normalizados = mifu.normalizar(datos_ordenados, 'minmax')
datos_divididos = mifu.dividir_datos(datos_normalizados, 0.25, prediccion)
[[x_entrenamiento, e_entrenamiento], [x_prueba, e_prueba]] = datos_divididos

plt.style.use('dark_background')
plt.figure(figsize=(16, 3))
for i in range(6):
    plt.plot(np.concatenate(datos_ordenados[0][:,(i*4096):((i+1)*4096)]), label = 'Canal ' + str(i+1))
plt.title("Valores de EEG, sujeto " + str(sujeto))
plt.xlabel("Sample")
plt.ylabel("Magnitud")
plt.legend()
plt.xlim(-100, np.concatenate(datos_ordenados[0][:,:4096]).shape[0] + 100)
plt.show()

mifu.graficar(datos, 1)
plt.plot(np.concatenate(datos_normalizados[0][:,0]))
plt.show()
plt.plot(datos_normalizados)

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

####################################################################################
# Probemos hacer una transformada de fourier discreta:
####################################################################################
x = []
y = []
equiespaciado = 1
L = 1
N = 100
if equiespaciado == 1:
    for i in range(N):
        x.append(i*L/N)
        y.append(random.random())
    wmax = N
elif equiespaciado == 0:
    for i in range(N):
        x.append(random.random())
        y.append(random.random())
    x = sorted(x)
    dist = []
    for i in range(len(x) - 1):
        dist.append(x[i + 1] - x[i])
    L0 = min(dist)
    L = (max(x) - min(x)) + L0
    wmax = L/L0
# y = []
# for i in range(N):
    if i > N/2 and i < 5*N/8 or i < N/5:
        y.append(1)
    else:
        y.append(0)

c = 2*math.pi/L
factor = 1
div = 1
Y = []
for k in range(int(wmax*factor*div)):
    R = 0
    I = 0
    for j in range(N):
        R += y[j]*math.cos(-c*k/div*x[j])
        I += y[j]*math.sin(-c*k/div*x[j])
    Y.append(complex(R,I))

n = int(factor*20*(N + 1))
xp = []
for i in range(n):
    xp.append(min(x) + L*i/n)
funcion = np.zeros(n)
for i in range(len(Y)):
    sumx = []
    for j in xp:
        sumx.append(Y[i]*(math.cos(c*i/div*j) + complex(0,1)*math.sin(c*i/div*j))/(N*factor*div))
    funcion = funcion + np.array(sumx)
    #plt.plot(xp, sumx)

plt.plot(xp, funcion, label = 'f = 0, 1, ... ' + str(factor*int(wmax - 1)), lw = 1)
plt.scatter(np.array(x), np.array(y), color = 'white')
mifu.grafico_oscuro('Tiempo', 'Valor', 'Datos y su reconstruccion mediante transformada de fourier')
plt.show()

plt.vlines(np.array(range(len(Y))) - L/(10*N), ymin = 0, ymax = np.abs(Y), label = '|F(w)|', color = 'blue')
plt.vlines(np.array(range(len(Y))) + L/(10*N), ymin = 0, ymax = np.arctan(np.imag(Y)/np.real(Y)), label = 'Fase (w)', color = 'red')
mifu.grafico_oscuro('Frecuencia', 'Coeficiente', 'Representacion en el espacio de frecuencia de los datos')
plt.grid(True, lw = 0.1)
plt.show()

# Filtrando
# Aca la idea es eliminar todas aquellas frecuencias menor a un umbral, es decir, que su aporte no sea significativo
umbral = 2
Yf = []
funcion_nueva = np.zeros(n)
for i in range(int(len(Y))):
    if np.abs(Y)[i] >= umbral:
        Yf.append((Y)[i])
        sumx = []
        for j in xp:
            sumx.append(Y[i]*(math.cos(c*i/div*j) + complex(0,1)*math.sin(c*i/div*j))/(N*factor*div))
        funcion_nueva = funcion_nueva + np.array(sumx)
        #plt.plot(xp, sumx)
    else:
        Yf.append(complex(0, 0))

# Reconstruyamos la señal
yf = np.zeros(len(x))
for i in range(len(Yf)):
    sumx = []
    for j in x:
        sumx.append(Yf[i]*(math.cos(c*i/div*j) + complex(0,1)*math.sin(c*i/div*j))/(N*factor*div))
    yf = yf + np.array(sumx)

plt.scatter(np.array(x), np.array(y), color = 'white')
plt.scatter(np.array(x), np.array(yf), color = 'red')
mifu.grafico_oscuro('Tiempo', 'Valor', 'Datos y su reconstruccion por filtrado')
plt.show()

plt.plot(xp, funcion, label = 'f = 0, 1, ... ' + str(factor*int(wmax - 1)), lw = 1, color = 'red')
plt.plot(xp, funcion_nueva, label = 'f >= ' + str(umbral), lw = 1, color = 'green')
plt.scatter(np.array(x), np.array(y), color = 'white')
mifu.grafico_oscuro('Tiempo', 'Valor', 'Datos, su reconstruccion y filtrado mediante transformada de fourier')
plt.show()

plt.vlines(np.array(range(len(Yf))) - L/(10*N), ymin = 0, ymax = np.abs(Yf), label = 'Componente real', color = 'blue')
plt.vlines(np.array(range(len(Yf))) + L/(10*N), ymin = 0, ymax = np.arctan(np.imag(Yf)/np.real(Yf)), label = 'Componente imaginaria', color = 'red')
mifu.grafico_oscuro('Frecuencia', 'Coeficiente', 'Representacion en el espacio de frecuencia de los datos')
plt.grid(True, lw = 0.1)
plt.show()

####################################################################################
# Quiero ver el espacio de frecuencias de los datos:
####################################################################################
fm = 1024
T = 4
N = T*fm

sujeto = 1

datos = mifu.extraer(sujeto)
datos_ordenados = mifu.ordenar(datos)
datos_normalizados = mifu.normalizar(datos_ordenados)

prediccion = ['modalidad', 'estimulo', 'artefacto'].index('estimulo')
bloque = random.randint(0, len(datos_normalizados[0]))
canal = 1
y = datos_normalizados[0][bloque - 1,:,canal - 1]
etiqueta = datos_normalizados[1][bloque - 1,prediccion]
print(etiqueta)

plt.scatter(range(len(y)), y, linewidths = 0)
mifu.grafico_oscuro('Muestras', 'Registro', 'Canal ' + str(canal) + ', sujeto ' + str(sujeto) + ', bloque ' + str(bloque))
plt.show()

fft_valores = np.fft.fft(y)
f = np.fft.fftfreq(N, d = 1/fm)
magnitud = np.abs(fft_valores)
fase = np.angle(fft_valores)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(f[:N // 2], magnitud[:N // 2])
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud')
plt.title('Transformada de Fourier - Magnitud')
plt.subplot(2, 1, 2)
plt.plot(f[:N // 2], fase[:N // 2])
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Fase (radianes)')
plt.title('Transformada de Fourier - Fase')
plt.tight_layout()
plt.show()



####################################################################################
# Transformada de fourier de tiempo corto:
####################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

def compute_stft(signal, fs=1024, window='hann', nperseg=256, noverlap=128):
    """
    Calcula la STFT de una señal EEG.
    
    Parámetros:
    - signal: np.array, señal de EEG a analizar
    - fs: int, frecuencia de muestreo (Hz)
    - window: str, tipo de ventana
    - nperseg: int, tamaño de cada segmento
    - noverlap: int, solapamiento entre ventanas
    
    Retorna:
    - f: frecuencias
    - t: tiempos
    - Zxx: matriz de STFT (espectrograma complejo)
    """
    f, t, Zxx = stft(signal, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    return f, t, Zxx

def plot_stft(f, t, Zxx):
    """Grafica el espectrograma de la STFT."""
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.colorbar(label='Magnitud')
    plt.ylabel('Frecuencia (Hz)')
    plt.xlabel('Tiempo (s)')
    plt.title('STFT - Espectrograma')
    plt.show()


sujeto = random.randint(1, 15)
datos = mifu.extraer(sujeto)
datos_ordenados = mifu.ordenar(datos)
fs = 1024
dato = random.randint(0, datos_ordenados[0].shape[0] - 1)
canal = random.randint(0, 5)
    
f, t_stft, Zxx = compute_stft(datos_ordenados[0][dato][canal*4096:(canal + 1)*4096], fs = fs)
plot_stft(f, t_stft, Zxx)