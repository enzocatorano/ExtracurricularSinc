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

######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
# esto es desde que escribi 'buenas_practicas.py'

import buenas_practicas as mbp
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

semilla = 42
torch.manual_seed(semilla)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(semilla)

print("Generando datos sintéticos...")
num_muestras = 500
num_caracteristicas = 20
num_clases = 3

# Datos de entrenamiento
X_train = torch.randn(num_muestras, num_caracteristicas)
# Crear etiquetas que tengan alguna relación (simple) con los datos
# Por ejemplo, si la suma de las primeras 5 características es positiva, clase 0, si es muy negativa clase 1, etc.
sum_features = X_train[:, :5].sum(dim=1)
y_train_list = []
for val in sum_features:
    if val > 1:
        y_train_list.append(0)
    elif val < -1:
        y_train_list.append(1)
    else:
        y_train_list.append(2)
y_train = torch.tensor(y_train_list, dtype=torch.long)


# Datos de validación
X_val = torch.randn(num_muestras // 2, num_caracteristicas)
sum_features_val = X_val[:, :5].sum(dim=1)
y_val_list = []
for val in sum_features_val:
    if val > 1:
        y_val_list.append(0)
    elif val < -1:
        y_val_list.append(1)
    else:
        y_val_list.append(2)
y_val = torch.tensor(y_val_list, dtype=torch.long)

# Crear DataLoaders
batch_size = 32
dataset_train = TensorDataset(X_train, y_train)
cargador_entrenamiento = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

dataset_val = TensorDataset(X_val, y_val)
cargador_validacion = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
print("Datos y DataLoaders listos.")

# 2. Instanciar MLP
print("\nInstanciando MLP...")
arquitectura_mlp = [num_caracteristicas, 64, 32, num_clases] # Entrada, ocultas, salida
modelo_mlp = mbp.MLP(
    arq=arquitectura_mlp,
    func_act='relu', # o ['relu', 'relu', 'identity'] si la última capa no lleva activación antes de CrossEntropy
    usar_batch_norm=True,
    dropout=0.2,
    metodo_init_pesos=nn.init.xavier_uniform_,
    semilla=semilla
)
print("Modelo MLP creado:")
# print(modelo_mlp) # __str__ ya está implementado

# 3. Instanciar Entrenador
print("\nInstanciando Entrenador...")
# El optimizador se puede definir aquí o dejar que Entrenador lo cree por defecto
optimizador_adam = optim.Adam(modelo_mlp.parameters(), lr=0.001)
funcion_perdida = nn.CrossEntropyLoss() # Adecuada para clasificación multiclase con salida de logits

entrenador = mbp.Entrenador(
    modelo=modelo_mlp,
    optimizador=optimizador_adam,
    func_perdida=funcion_perdida,
    device='cuda' if torch.cuda.is_available() else 'cpu', # 'cuda' o 'cpu'
    parada_temprana=5, # Número de épocas sin mejora antes de parar
    log_dir='runs/prueba_mlp_entrenador' # Directorio para logs de TensorBoard
)
print("Entrenador listo.")

# 4. Entrenar el modelo
print("\nIniciando entrenamiento...")
num_epocas = 100
entrenador.ajustar(
    cargador_entrenamiento=cargador_entrenamiento,
    cargador_validacion=cargador_validacion,
    epocas=num_epocas
)
print("Entrenamiento completado.")
print(f"Puedes ver los logs de TensorBoard ejecutando: tensorboard --logdir={entrenador.escritor.log_dir}")

# 5. Evaluación simple (opcional, ya que Entrenador lo hace en validación)
print("\nEvaluando modelo en datos de validación (después del entrenamiento)...")
modelo_mlp.eval() # Poner el modelo en modo evaluación
correctas_final = 0
total_final = 0
perdida_final_total = 0.0

# Usar el dispositivo correcto para la evaluación
device_eval = entrenador.device 
modelo_mlp.to(device_eval)

with torch.no_grad():
    for x_batch, y_batch in cargador_validacion:
        x_batch, y_batch = x_batch.to(device_eval), y_batch.to(device_eval)
        
        salidas = modelo_mlp(x_batch)
        perdida = funcion_perdida(salidas, y_batch)
        perdida_final_total += perdida.item() * x_batch.size(0)
        
        _, predicciones = torch.max(salidas.data, 1)
        total_final += y_batch.size(0)
        correctas_final += (predicciones == y_batch).sum().item()

precision_final = 100 * correctas_final / total_final if total_final > 0 else 0
perdida_promedio_final = perdida_final_total / total_final if total_final > 0 else float('inf')
print(f'Pérdida en el conjunto de validación: {perdida_promedio_final:.4f}')
print(f'Precisión en el conjunto de validación: {precision_final:.2f}%')

# Ejemplo de cómo obtener una predicción para una nueva muestra
print("\nEjemplo de predicción para una nueva muestra:")
nueva_muestra = torch.randn(1, num_caracteristicas).to(device_eval) # Crear una muestra y moverla al dispositivo
modelo_mlp.eval()
with torch.no_grad():
    prediccion_nueva = modelo_mlp(nueva_muestra)
    _, clase_predicha = torch.max(prediccion_nueva.data, 1)
    print(f"Logits de predicción: {prediccion_nueva.cpu().numpy()}")
    print(f"Clase predicha: {clase_predicha.cpu().item()}")

for nombre, param in modelo_mlp.named_parameters():
    print(nombre, param.shape)

#############################################################################################################################
# probando mne

import numpy as np
import mne
import matplotlib.pyplot as plt

# Generate synthetic EEG data: 2 channels, 10 seconds at 100 Hz sampling
sfreq = 100  # sampling frequency
times = np.arange(0, 10, 1/sfreq)
# Create a 10 Hz sine wave on channel 1, 20 Hz on channel 2
sin1 = np.sin(2 * np.pi * 10 * times)
sin2 = np.sin(2 * np.pi * 20 * times)
data = np.vstack([sin1, sin2])

# Create MNE Raw object
info = mne.create_info(ch_names=['Ch1', 'Ch2'], sfreq=sfreq, ch_types=['eeg', 'eeg'])
raw = mne.io.RawArray(data, info)

# Plot time series (first 5 seconds) - Esto parece funcionar para ti
plt.figure(figsize=(16, 4))
plt.plot(raw.times[:5*sfreq], raw.get_data()[0, :5*sfreq]) # Usar get_data() es más seguro que _data
plt.title("Synthetic EEG Signal (Ch1, 10 Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# plotea la PSD de todos los canales hasta 50 Hz
raw.plot_psd(fmax=50, n_fft=256)

##############################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Synthetic EEG data
sfreq = 100  # sampling frequency
times = np.arange(0, 10, 1/sfreq)
sin1 = np.sin(2 * np.pi * 10 * times)
sin2 = np.sin(2 * np.pi * 20 * times)
data = np.vstack([sin1, sin2])

# FFT calculation
N = data.shape[1]
freqs_fft = np.fft.rfftfreq(N, d=1/sfreq)
fft_vals = np.fft.rfft(data, axis=1)
fft_power = (np.abs(fft_vals) ** 2) / N

# Welch PSD calculation for comparison
nperseg = 256
noverlap = nperseg // 2
freqs_welch, psd_welch = welch(data, fs=sfreq, nperseg=nperseg, noverlap=noverlap, axis=1)

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

for idx, ax in enumerate(axs):
    ax.stem(freqs_fft, fft_power[idx], label='FFT Power Spectrum')
    ax.stem(freqs_welch, 100*psd_welch[idx], label='Welch PSD', linefmt = 'orange')
    ax.set_title(f'Channel {idx+1}')
    ax.set_ylabel('Power')
    ax.legend()

axs[-1].set_xlabel('Frequency (Hz)')
plt.tight_layout()
plt.show()


for i in range(len(freqs_fft)):
    print(freqs_fft[i], fft_power[0][i])

##############################################################################################################################
# pruebo con datos reales

import numpy as np
import scipy.io as sio
import mne
import matplotlib.pyplot as plt
from scipy.signal import stft

# 1. Cargar datos
ruta = '../Base_de_Datos_Habla_Imaginada/S01/S01_EEG.mat'
mat = sio.loadmat(ruta, squeeze_me=True, struct_as_record=False)
eeg = np.array(mat['EEG'])

# 2. Separar datos EEG (primeras 24576 columnas)
eeg_data = eeg[:, :24576]

# 3. Función para dividir en 6 canales
n_samples = 4096
def dividir_por_canales(muestra):
    return np.vstack([
        muestra[0*n_samples : 1*n_samples],
        muestra[1*n_samples : 2*n_samples],
        muestra[2*n_samples : 3*n_samples],
        muestra[3*n_samples : 4*n_samples],
        muestra[4*n_samples : 5*n_samples],
        muestra[5*n_samples : 6*n_samples],
    ])

# 4. Preparar la primera muestra
trial = dividir_por_canales(eeg_data[0])[np.newaxis, ...]

# Parámetros
sfreq = 1024.0
freqs = np.arange(0.001, 65, 1)
n_cycles = freqs / 2.

# 5. TFR con wavelets Morlet
power = mne.time_frequency.tfr_array_morlet(
    trial, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles, output='power'
)

# 6. STFT
signal = trial[0, 0]
f_stft, t_stft, Zxx = stft(signal, fs=sfreq, nperseg=256, noverlap=128)

# 7. Ploteo conjunto
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
# TFR Morlet
ax = axs[0]
im1 = ax.imshow(
    power[0, 0], aspect='auto', origin='lower', interpolation = 'nearest', cmap = 'cividis',
    extent=[0, trial.shape[-1]/sfreq, freqs[0], freqs[-1]]
)
ax.set_xlabel("Tiempo (s)")
ax.set_ylabel("Frecuencia (Hz)")
ax.set_title("TFR Morlet - Canal F3")
fig.colorbar(im1, ax=ax, label="Potencia")
# STFT
lim = 15
ax = axs[1]
pcm = ax.pcolormesh(
    t_stft, f_stft[:lim], np.abs(Zxx[:lim,:]), shading='nearest', cmap='cividis'
)
ax.set_xlabel("Tiempo (s)")
ax.set_ylabel("Frecuencia (Hz)")
ax.set_title("STFT - Canal F3")
fig.colorbar(pcm, ax=ax, label="Magnitud")

plt.tight_layout()
plt.show()

################################################################################################################################################
# mapa topografico?
n_trials = eeg_data.shape[0]
dato = np.random.randint(0, n_trials)
trials = np.stack([dividir_por_canales(eeg_data[i]) for i in range(n_trials)])

# 4. Crear objeto EpochsArray de MNE
sfreq = 1024.0
ch_names = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4']
ch_types = ['eeg'] * len(ch_names)
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
epochs = mne.EpochsArray(trials, info, tmin=0.0)

# 5. Asignar montaje estándar 10-20
montage = mne.channels.make_standard_montage('standard_1020')
epochs.set_montage(montage)

# 6. Calcular PSD multitaper en banda alfa (8–12 Hz) usando psd_array_multitaper
#    psd_array_multitaper recibe data shape (n_epochs, n_channels, n_times)
from mne.time_frequency import psd_array_multitaper
psds, freqs = psd_array_multitaper(
    epochs.get_data(), sfreq=sfreq, fmin=0, fmax=12, bandwidth=4.0, adaptive=True, normalization='full', verbose=False
)
# psds shape: (n_epochs, n_channels, n_freqs)
# 7. Promediar en frecuencia y en ensayos
psd_mean = psds.mean(axis=0).mean(axis=1)  # (n_channels,)

# 8. Graficar mapa topográfico de la potencia alfa media
fig, ax = plt.subplots()
mne.viz.plot_topomap(psd_mean, epochs.info, axes=ax, contours=0)
ax.set_title('Potencia media α (8–12 Hz) - Multitaper')
plt.show()

################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
# pruebo la nueva forma de cargar los datos

import buenas_practicas as mbp
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

sujeto = 12
prediccion = 'estimulo'
datos = mbp.DataSetEEG(sujeto)
datos.graficar_canales(np.random.randint(0, len(datos)))
entrenamiento, validacion, prueba = datos.particionar(0.7, True, 17)
datos.normalizar()
datos.dejar_etiqueta(prediccion)

# pruebo una clasificacion usando mbp.MLP
arq = [entrenamiento[0][0].shape[0], 256, 256, 11]
func_act = 'relu'
modelo = mbp.MLP(arq, func_act = func_act)
entrenador = mbp.Entrenador(
                            modelo = modelo,
                            optimizador = optim.Adam(modelo.parameters(), lr = 0.00001, weight_decay = 0.0001),
                            func_perdida = nn.CrossEntropyLoss(),
                            device = 'cuda' if torch.cuda.is_available() else 'cpu',
                            parada_temprana = 100,
                            log_dir = 'runs/prueba_dataset_crudo_mlp'
                            )
batch_size = 31
cargador_entrenamiento = DataLoader(entrenamiento, batch_size = batch_size, shuffle = True, drop_last = False)
cargador_validacion = DataLoader(validacion, batch_size = batch_size, shuffle = False)

entrenador.ajustar(cargador_entrenamiento, cargador_validacion, epocas = 1000)

# probemos el evaluador
cargador_prueba = DataLoader(prueba, batch_size = 1, shuffle = False)
evaluador = mbp.Evaluador(modelo = modelo, clases = prediccion)

evaluador.matriz_confusion(cargador_prueba)
#evaluador.reporte(cargador_prueba)

################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
# FFT y bandas

import buenas_practicas as mbp
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import matplotlib.pyplot as plt

sujeto = 1
prediccion = 'estimulo'
datos = mbp.DataSetEEG(sujeto)
datos.graficar_canales(np.random.randint(0, len(datos)))
x = datos[:][0]
y = datos[:][1]

# quiero graficar el espacio de frecuencias de la señal
sfreq = 1024
n_samples = 4096
n_canal = 1

# Tomar una muestra aleatoria
idx_muestra = np.random.randint(0, len(datos))
muestra_eeg = x[idx_muestra][(n_canal-1)*n_samples:(n_canal)*n_samples].numpy() # Convertir a numpy

# Calcular FFT para cada canal
fft_datos = np.fft.rfft(muestra_eeg)
freqs = np.fft.rfftfreq(n_samples, d=1/sfreq)

# Definir bandas de frecuencia (ejemplo)
bandas = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta': (12, 30),
    'Gamma': (30, 60)
}

# Reconstruir la señal para cada banda
reconstrucciones = {}
for nombre_banda, (fmin, fmax) in bandas.items():
    # Crear una copia de los coeficientes FFT
    fft_filtrada = np.copy(fft_datos)

    # Poner a cero los coeficientes fuera de la banda de interés
    fft_filtrada[(freqs < fmin) | (freqs > fmax)] = 0

    # Realizar la iFFT para reconstruir la señal en el dominio del tiempo
    senal_reconstruida = np.fft.irfft(fft_filtrada, n=n_samples)
    reconstrucciones[nombre_banda] = senal_reconstruida

# ahora quiero graficar la señal original arriba, y hacia abajo
# en figuras diferentes, cada reconstruccion de la misma segun cada
# banda de frecuencia
plt.style.use('dark_background')
fig, axs = plt.subplots(len(bandas) + 1, 1, figsize=(12, 1.5 * (len(bandas) + 1)), sharex=True)

# Graficar señal original
axs[0].plot(np.arange(n_samples)/sfreq, muestra_eeg, color='dodgerblue')
axs[0].set_title(f'Señal Original - Sujeto {sujeto}, Muestra {idx_muestra}, Canal {n_canal}')
axs[0].set_ylabel('Amplitud')
axs[0].grid(linewidth=0.2, linestyle='--')

# Graficar reconstrucciones por banda
for i, (nombre_banda, senal_reconstruida) in enumerate(reconstrucciones.items()):
    axs[i+1].plot(np.arange(n_samples)/sfreq, senal_reconstruida, color='orange')
    axs[i+1].set_title(f'{nombre_banda} ({bandas[nombre_banda][0]}-{bandas[nombre_banda][1]} Hz)')
    axs[i+1].set_ylabel('Amplitud')
    axs[i+1].grid(linewidth=0.2, linestyle='--')

axs[-1].set_xlabel('Tiempo (s)')
plt.tight_layout()
plt.show()

################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
# stft y productos internos

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import buenas_practicas as mbp

sujeto = 1
datos = mbp.DataSetEEG(sujeto)
prediccion = 'estimulo'

ensayo = np.random.randint(0, len(datos))
canal = np.random.randint(0,6)
sample = np.random.randint(0, 4096)/1024

x = datos[:][0]
x = x[ensayo][canal*4096:(canal + 1)*4096]
y = datos[:][1]
t = np.arange(0, 4, 1/1024)
desv = 0.005
ventana = np.exp(-((t - sample)**2)/desv)
sin1 = np.sin(2 * np.pi * 2 * t)
sin2 = np.sin(2 * np.pi * 20 * t)

fig, ax = plt.subplots(6, 1, figsize = (12, 9))
ax[0].plot(t, x, color = 'dodgerblue')
ax[0].set_title(f'Señal Original - Sujeto {sujeto}, Ensayo {ensayo}, Canal {canal}')
ax[0].grid(linewidth = 0.2, linestyle = '--')
ax[0].set_xlim(-0.05, 4.15)
ax[0].set_xticks(np.arange(0, 4.25, 0.25))
ax[1].set_title(f'Ventana Gaussiana centrada en {sample:.2f}s')
ax[1].grid(linewidth = 0.2, linestyle = '--')
ax[1].set_xlim(-0.05, 4.15)
ax[1].set_xticks(np.arange(0, 4.25, 0.25))
ax[1].tick_params(axis = 'x', which = 'both', bottom = False, top = False)
ax[2].set_title('Onda Senoidal de 2 Hz')
ax[2].grid(linewidth = 0.2, linestyle = '--')
ax[2].set_xlim(-0.05, 4.15)
ax[2].set_xticks(np.arange(0, 4.25, 0.25))
ax[2].tick_params(axis = 'x', which = 'both', bottom = False, top = False)
ax[3].set_title('Producto de la señal ventaneada y la onda de 2 Hz')
ax[3].grid(linewidth = 0.2, linestyle = '--')
ax[3].set_xlim(-0.05, 4.15)
ax[3].set_xticks(np.arange(0, 4.25, 0.25))
ax[3].tick_params(axis = 'x', which = 'both', bottom = False, top = False)
ax[4].set_title('Onda Senoidal de 20 Hz')
ax[4].grid(linewidth = 0.2, linestyle = '--')
ax[4].set_xlim(-0.05, 4.15)
ax[4].set_xticks(np.arange(0, 4.25, 0.25))
ax[4].tick_params(axis = 'x', which = 'both', bottom = False, top = False)
ax[5].set_title('Producto de la señal ventaneada y la onda de 20 Hz')
ax[5].grid(linewidth = 0.2, linestyle = '--')
ax[5].set_xlim(-0.05, 4.15)
ax[5].set_xticks(np.arange(0, 4.25, 0.25))
ax[5].set_xlabel('Tiempo [s]')

ax[1].plot(t, ventana*float(x.abs().max()), color = 'white', linewidth = 0.5)
ax[1].plot(t, -ventana*float(x.abs().max()), color = 'white', linewidth = 0.5)
ax[1].plot(t, x*ventana, color = 'dodgerblue')
ax[2].plot(t, sin1, color = 'dodgerblue')
ax[3].plot(t, x*ventana*sin1, color = 'dodgerblue')
ax[3].fill_between(t, x*ventana*sin1, 0, label = f'{(x*ventana*sin1).sum()}', color = 'dodgerblue', alpha = 0.5)
ax[3].legend()
ax[4].plot(t, sin2, color = 'dodgerblue')
ax[5].plot(t, x*ventana*sin2, color = 'dodgerblue')
ax[5].fill_between(t, x*ventana*sin2, 0, label = f'{(x*ventana*sin2).sum()}', color = 'dodgerblue', alpha = 0.5)
ax[5].legend()
plt.tight_layout()
plt.show()

################################################################################################################################################
# quiero mostrar la fft de una señal que se ventanea con una ventana cada vez mas grande

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import buenas_practicas as mbp

sujeto = 1
datos = mbp.DataSetEEG(sujeto)
prediccion = 'estimulo'

ensayo = np.random.randint(0, len(datos))
canal = np.random.randint(0,6)
sample = np.random.randint(0, 4096)/1024

x = datos[:][0]
x = x[ensayo][canal*4096:(canal + 1)*4096]
t = np.arange(0, 4, 1/1024)

ventanas = [1, 0.1, 0.01, 0.001] # Desviaciones estándar para ventanas Gaussianas
n_fft = 4096 # Tamaño de la FFT

# quiero que en la columna de la izquierda esten la señal orignial y sus ventaneos
# y en la derecha los respectivos fft
fig, axs = plt.subplots(len(ventanas) + 1, 2, figsize=(16, 1.5 * (len(ventanas) + 1)), sharex='col')

# Graficar señal original en la primera fila, primera columna
axs[0, 0].plot(t, x, color='dodgerblue')
axs[0, 0].set_title(f'Señal Original - Sujeto {sujeto}, Ensayo {ensayo}, Canal {canal}')
axs[0, 0].grid(linewidth=0.2, linestyle='--')
axs[0, 0].set_xlim(-0.05, 4.15)
axs[0, 0].set_ylim(1.1*float(x.min()), 1.1*float(x.max()))
axs[0, 0].set_xticks(np.arange(0, 4.25, 0.25))
axs[0, 0].tick_params(axis='x', which='both', bottom=False, top=False)

# Calcular y graficar FFT de la señal original (columna derecha, primera fila)
sfreq = 1024
fft_vals_orig = np.fft.rfft(x.numpy(), n=n_fft)
freqs_orig = np.fft.rfftfreq(n_fft, d=1/sfreq)
magnitud_orig = np.abs(fft_vals_orig)
axs[0, 1].plot(freqs_orig, magnitud_orig, color='orange')
axs[0, 1].set_title('Magnitud FFT de Señal Original')
axs[0, 1].grid(linewidth=0.2, linestyle='--')
axs[0, 1].set_xlim(0, 64) # Limitar a una banda de frecuencia relevante
axs[0, 1].set_xticks(np.arange(0, 65, 5))
axs[0, 1].set_ylim(-0.1*magnitud_orig.max(), 1.1*magnitud_orig.max())
axs[0, 1].tick_params(axis='x', which='both', bottom=False, top=False)


# Graficar señales ventaneadas y sus FFTs en las filas siguientes
for i, desv in enumerate(ventanas):
    ventana = np.exp(-((t - sample)**2)/desv)
    senal_ventaneada = x * ventana

    # Calcular FFT
    fft_vals = np.fft.rfft(senal_ventaneada, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1/sfreq)
    magnitud = np.abs(fft_vals)

    # Graficar señal ventaneada en la fila siguiente, primera columna
    axs[i+1, 0].plot(t, senal_ventaneada, color='dodgerblue')
    axs[i+1, 0].plot(t, ventana*float(x.abs().max()), color='white', linewidth=0.5)
    axs[i+1, 0].plot(t, -ventana*float(x.abs().max()), color='white', linewidth=0.5)
    axs[i+1, 0].set_title(f'Señal Ventaneada (desv={desv})')
    axs[i+1, 0].grid(linewidth=0.2, linestyle='--')
    axs[i+1, 0].set_xlim(-0.05, 4.15)
    axs[i+1, 0].set_ylim(1.1*float(x.min()), 1.1*float(x.max()))
    axs[i+1, 0].set_xticks(np.arange(0, 4.25, 0.25))
    if i < len(ventanas) - 1:
        axs[i+1, 0].tick_params(axis='x', which='both', bottom=False, top=False)
    else:
        axs[i+1, 0].set_xlabel('Tiempo [s]')

    # Graficar FFT en la fila siguiente, segunda columna
    axs[i+1, 1].plot(freqs, magnitud, color='orange')
    axs[i+1, 1].set_title(f'Magnitud FFT (desv={desv})')
    axs[i+1, 1].grid(linewidth=0.2, linestyle='--')
    axs[i+1, 1].set_xlim(0, 64) # Limitar a una banda de frecuencia relevante
    axs[i+1, 1].set_xticks(np.arange(0, 65, 5))
    if i < len(ventanas) - 1:
        axs[i+1, 1].tick_params(axis='x', which='both', bottom=False, top=False)
    else:
        axs[i+1, 1].set_xlabel('Frecuencia [Hz]')

plt.tight_layout()
plt.show()