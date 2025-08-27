# Vamos a explorar un poco el dataset que tienen en el sinc.

#    Data relevante:
#       Estimulo indicatorio visual y auditivo de 2 segundos de duracion.
#       Una sola sesion de medidas por sujeto, 15 sujetos.
#       El habla imaginada consistia en 2 grupos de palabras en español:
#           1. Las vocales: "a", "e", "i", "o" y "u".
#           2. Comandos: "arriba", "abajo", "izquierda", "derecha", "adelante" y "atras".
#       A cada sujeto se le registro el habla imaginada 50 veces por cada palabra.
#       Para poder contrastar, 10 de esas 50 veces, el registro fue de habla pronunciada e
#   imaginada contando tambien con un microfono.
#       La instancia de imaginacion/pronunciacion (indicado en pantalla) de la palabra dura
#   4 segundos, en donde:
#           1. Si es una vocal, se hace continuamente durante los 4 segundos.
#           2. Si es un comando, 3 clicks audibles durante el periodo indican cuando hacerlo.
#       Durante la instancia de i/p los sujetos no se mueven, tragan ni parpadean.
#       Luego de esta, le sigue un periodo de descanso de 4 segundos donde si pueden.
#       Las palabras se muestran para ambos bloques, aleatoriamente.
#       Las mediciones de los bloques de vocales y comandos se hacen por separado, en sesiones
#   de 2 (10 palabras) y 2.43 (12 palabras) minutos de duracion respectivamente.
#      

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
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.signal import butter, filtfilt
from scipy.signal import stft

def importar_funciones (): 
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
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    from scipy.signal import butter, filtfilt
    from scipy.signal import stft

# fuera de uso
def arreglar_eeg (ruta_entrada):
    importar_funciones()
    ruta_salida = os.path.join('Datos', 'Solo_ordenados')
    if not os.path.exists(ruta_salida):
        os.makedirs(ruta_salida)
    for k in range(1,16):
        sujeto = str(k)
        if len(sujeto) == 1:
            sujeto = '0' + sujeto
        print('Escribiendo sujeto ' + sujeto)
        ruta_datos = os.path.join(ruta_salida, sujeto + '_datos.npy')
        ruta_etiquetas = os.path.join(ruta_salida, sujeto + '_etiquetas.npy')
        if not (os.path.exists(ruta_datos) and os.path.exists(ruta_etiquetas)):
            ruta_entrada1 = ruta_entrada + '/Base de Datos Habla Imaginada/S' + sujeto + '/S' + sujeto + '_EEG.mat'
            data = scipy.io.loadmat(ruta_entrada1)
            EEG = data['EEG']
            F3 = []
            F4 = []
            C3 = []
            C4 = []
            P3 = []
            P4 = []
            modalidad = []
            estimulo = []
            artefacto = []
            n = 4096
            for i in EEG:
                for j in range(0, n):
                    F3.append(i[j])
                    F4.append(i[j + n])
                    C3.append(i[j + 2*n])
                    C4.append(i[j + 3*n])
                    P3.append(i[j + 4*n])
                    P4.append(i[j + 5*n])
                modalidad.append(int(i[6*n]) - 1)
                estimulo.append(int(i[6*n + 1]) - 1)
                artefacto.append(int(i[6*n + 2]) - 1)
            datos = np.array([F3, F4, C3, C4, P3, P4])
            etiquetas = np.array([modalidad, estimulo, artefacto])
            np.save(ruta_datos, datos)
            np.save(ruta_etiquetas, etiquetas)
            print('Datos del sujeto extraidos correctamente.')
        else:
            print('Los datos de este sujeto ya fueron extraidos, no se reescribiran.')
    print('Extraccion total finalizada')
    return

def extraer (sujeto, ruta = os.path.dirname(os.getcwd())):
    importar_funciones()
    sujeto = str(sujeto)
    if len(sujeto) == 1:
        sujeto = '0' + sujeto
    ruta = ruta + '/Base_de_Datos_Habla_Imaginada/S' + sujeto + '/S' + sujeto + '_EEG.mat'
    data = scipy.io.loadmat(ruta)
    EEG = data['EEG']
    return(EEG)

# fuera de uso
def graficar (data, mod = 0):
    importar_funciones()
    n = 4096
    modalidad_posible = ['Imaginada', 'Pronunciada']
    estimulo_posible = ['A', 'E', 'I', 'O', 'U', 'Arriba', 'Abajo', 'Adelante', 'Atras', 'Derecha', 'Izquierda']
    artefacto_posible = ['Ninguno', 'Parpadeo']
    F3 = []
    F4 = []
    C3 = []
    C4 = []
    P3 = []
    P4 = []
    modalidad = []
    estimulo = []
    artefacto = []
    for i in data:
        for j in range(0, n):
            F3.append(i[j])
            F4.append(i[j + n])
            C3.append(i[j + 2*n])
            C4.append(i[j + 3*n])
            P3.append(i[j + 4*n])
            P4.append(i[j + 5*n])
        modalidad.append(int(i[6*n]) - 1)
        estimulo.append(int(i[6*n + 1]) - 1)
        artefacto.append(int(i[6*n + 2]) - 1)
    plt.plot(F3, label = 'F3')
    plt.plot(F4, label = 'F4')
    plt.plot(C3, label = 'C3')
    plt.plot(C4, label = 'C4')
    plt.plot(P3, label = 'P3')
    plt.plot(P4, label = 'P4')
    if mod != 0:
        colores = ['red', 'blue', 'green', 'purple', 'orange', 'saddlebrown',
                    'hotpink', 'dimgray', 'cyan', 'magenta', 'yellow']
        i = 0
        while i < len(estimulo):
            pos = estimulo[i]
            plt.axvspan(i*n, n*(i + 1), alpha = 0.15, color = colores[pos])
            i += 1
    # Modo oscuro
    ax = plt.gca()
    ax.set_facecolor('black')
    plt.gcf().set_facecolor('black')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.xlabel('Muestras', fontsize=12, color = 'white')
    plt.ylabel('Valor', fontsize=12, color = 'white')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=6, facecolor='black', edgecolor='white', labelcolor='white')
    plt.tight_layout()
    plt.show()

def ordenar (data):
    ''' Devuelve una lista de forma [datos, etiquetas], donde datos es un array de N mediciones (cambiando estimulo y modalidad).
     Cada una de las N mediciones esta dada como una concatenacion de los 6 canales de EEG que registraron los datos.
     Asi, el array de datos tiene dimension N x 24576, donde el segundo valor viene de tener 6 canales con 4096 samples (4 seg. a 1024 Hz).
     Las etiquetas son un array de dimension N x 3. N mediciones (igual que antes), con 3 etiquetas correspondientes a modalidad,
     estimulo y presencia de artefactos. '''
    importar_funciones()
    n = 4096
    datos = []
    etiquetas = []
    for i in data:
        bloque = np.array(i[0:6*n]).T
        etiquetas.append([i[6*n], i[6*n + 1], i[6*n + 2]])
        datos.append(bloque)
    datos = np.array(datos)
    etiquetas = np.array(etiquetas)
    return([datos, etiquetas])

def normalizar (data):
    ''' Realiza normalizacion min-max de forma global, es decir, respecto del valor más grande y más chico
     de entre todos los valores de la variable tipo array "datos" que arroja la funcion "ordenar". '''
    importar_funciones()
    datos = []
    max_global = np.max(data[0])
    min_global = np.min(data[0])
    for i in data[0]:
        datos.append((i - min_global) / (max_global - min_global))
    datos = np.array(datos)
    return([datos, data[1]])

def crear_butterworth(tipo, corte_bajo, corte_alto, fs, orden = 4): # crea el filtro
    nyquist = 0.5 * fs  # Frecuencia de Nyquist
    bajo = corte_bajo / nyquist
    alto = corte_alto / nyquist
    if tipo == 'band':
        b, a = butter(orden, [bajo, alto], btype = tipo)
    elif tipo == 'low':
        b, a = butter(orden, alto, btype = tipo)
    elif tipo == 'high':
        b, a = butter(orden, bajo, btype = tipo)
    return b, a

def filtrar_butterworth(data, tipo, corte_bajo, corte_alto, fs, orden = 4): # aplica el filtro
    b, a = crear_butterworth(tipo, corte_bajo, corte_alto, fs, orden = orden)
    datos_filtrados = np.empty_like(data[0])
    for i in range(data[0].shape[0]): # itera en cada medicion
        for j in range(6): # itera en cada canal concatenado
            datos_filtrados[i][j*4096:(j + 1)*4096] = filtfilt(b, a, data[0][i][j*4096:(j + 1)*4096])
    return([datos_filtrados, data[1]])

def contar_etiquetas (data, tipo = 'estimulo'):
    importar_funciones()
    x = ['modalidad', 'estimulo', 'artefacto'].index(tipo)
    posibles = [['Imaginada', 'Pronunciada'], ['A', 'E', 'I', 'O', 'U', 'Arriba', 'Abajo',
                 'Adelante', 'Atras', 'Derecha', 'Izquierda'], ['Ninguno', 'Parpadeo']]
    cantidad = [[0,0],[0,0,0,0,0,0,0,0,0,0,0,],[0,0]]
    indices = [[[], []], [[], [], [], [], [], [], [], [], [], [], []], [[], []]]
    j = 0
    for i in data[1][:,x]:
        cantidad[x][int(i) - 1] += 1
        indices[x][int(i) - 1].append(j)
        j += 1
    return([cantidad[x], indices[x]])

def dividir_datos(data, fraccion_entrenamiento, tipo, azar = 17):
    [cantidad, indices] = contar_etiquetas(data, tipo)
    chico = min(cantidad)
    ne = int(chico * fraccion_entrenamiento)
    print('Se usara el ' +  str(int(ne*len(cantidad)*100/sum(cantidad))) + '% para entrenar al modelo. Son ' + str(ne) + ' datos de c/u.')
    x = ['modalidad', 'estimulo', 'artefacto'].index(tipo)
    etiquetas_unicas = np.unique(data[1][:,x])
    datos_entrenamiento, etiquetas_entrenamiento, datos_prueba, etiquetas_prueba = [], [], [], []
    for i in etiquetas_unicas:
        datos_etiqueta = data[0][data[1][:,x] == i]
        etiq_etiqueta = data[1][:,x][data[1][:,x] == i]
        datos_etiqueta, etiq_etiqueta = shuffle(datos_etiqueta, etiq_etiqueta, random_state = azar)
        datos_entrenamiento.append(datos_etiqueta[:ne])
        etiquetas_entrenamiento.append(etiq_etiqueta[:ne])
        datos_prueba.append(datos_etiqueta[ne:])
        etiquetas_prueba.append(etiq_etiqueta[ne:])
    entrenamiento = [np.concatenate(datos_entrenamiento), np.concatenate(etiquetas_entrenamiento)]
    prueba = [np.concatenate(datos_prueba), np.concatenate(etiquetas_prueba)]
    entrenamiento[1] = (entrenamiento[1] - 1).astype(int)
    prueba[1] = (prueba[1] - 1).astype(int)
    return([entrenamiento, prueba])

def preparar_datos(data, f_entrenamiento = 0.2, etiqueta_objetivo = 1):
    """Divide datos y etiquetas en entrenamiento y prueba."""
    datos = data[0]
    etiquetas = data[1] - 1
    datos_entrenamiento, datos_prueba, etiquetas_entrenamiento, etiquetas_prueba = train_test_split(
        datos, etiquetas[:, etiqueta_objetivo], train_size = f_entrenamiento, stratify = etiquetas[:, etiqueta_objetivo]
    )
    return [[datos_entrenamiento, etiquetas_entrenamiento], [datos_prueba, etiquetas_prueba]]

def calcular_stft(signal, fs = 1024, window = 'hann', nperseg = 256, noverlap = 128):
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
    f, t, Zxx = stft(signal, fs = fs, window = window, nperseg = nperseg, noverlap = noverlap, )
    return f, t, Zxx

def convertir_a_stft(datos, fs = 1024, window = 'hann', nperseg = 256, noverlap = 128):
    f, t, matriz_prueba = calcular_stft(datos[0][0][0:fs*4], fs, window, nperseg, noverlap)
    reformados = np.zeros((datos[0].shape[0], 6, matriz_prueba.shape[0], matriz_prueba.shape[1])).astype(np.complex128())
    for i in range(datos[0].shape[0]):
        for j in range(6):
            f, t, Zxx = calcular_stft(datos[0][i][j*fs*4:(j+1)*fs*4], fs, window, nperseg, noverlap)
            reformados[i][j] = Zxx
    return [reformados, datos[1], [t, f]]

def calcular_potencia_bandas(matriz, f, bandas):
    '''
    Calcula la potencia de cada banda especificada para cada ventana temporal del espectrograma.
    '''
    salida = {nombre: [] for nombre in bandas.keys()}   # Hace un diccionario con las mismas llaves
    potencia_espectral = np.abs(matriz) **2
    for nombre, (f_min, f_max) in bandas.items():
        mascara = (f > f_min) & (f < f_max)
        salida[nombre] = np.sum(potencia_espectral[mascara, :], axis = 0)
    return salida

def convertir_a_potencia_bandas(datos, bandas):
    data = []
    for i in datos[0]:
        aux2 = []
        for j in i:
            aux = []
            dictpb = calcular_potencia_bandas(j, datos[2][1], bandas)
            for k in dictpb.values():
                aux.append(k)
            aux2.append(np.concatenate(np.array(aux)))
        data.append(np.concatenate(np.array(aux2)))
    data = np.array(data)
    return [data, datos[1]]

class MLP(nn.Module):
    def __init__(self, n_entrada, n_ocultas, n_salida, dropout_ratio = 0.1):
        super(MLP, self).__init__()
            
        capas = []
        j = n_entrada
        for i in n_ocultas:
            capas.append(nn.Linear(j, i))
            capas.append(nn.ReLU())
            capas.append(nn.Dropout(dropout_ratio))
            j = i
        capas.append(nn.Linear(j, n_salida))
        
        self.network = nn.Sequential(*capas)
    
    def forward(self, x):
        return self.network(x)
        
    def entrenar(modelo, epocas, data, etiq, plotear, tolerancia, lr = 0.001):
        datos = torch.from_numpy(data).float()
        etiquetas = torch.from_numpy(etiq).long()

        optimizador = optim.Adam(modelo.parameters(), lr)
        criterio = nn.CrossEntropyLoss()
        perdida = []
        vec_epoca = []
        minima_perdida = float('inf')
        contador = 0

        for epoch in range(epocas):
            modelo.train()
            optimizador.zero_grad()
                
            outputs = modelo(datos)
            loss = criterio(outputs, etiquetas)
                
            loss.backward()
            optimizador.step()
            
            valor_perdida = loss.item()
            if valor_perdida < minima_perdida - tolerancia:
                minima_perdida = valor_perdida
                contador = 0
            else:
                contador += 1

            if plotear == 1 and epoch % 5 == 0:
                perdida.append(loss.item())
                vec_epoca.append(epoch)

            if contador >= 50:
                print('Tras ' + str(epoch) + ' epocas, se ejcutó la parada temprana.')
                break

        if plotear == 1:
            plt.plot(vec_epoca, perdida, label = "Pérdida")
            plt.xlabel("Épocas", fontsize=12, color = 'white')
            plt.ylabel("Pérdida", fontsize=12, color = 'white')
            plt.title("Perdida por epocas", fontsize=24, color = 'white')
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

    def evaluar(modelo, data, etiq):
        datos = torch.from_numpy(data).float()
        etiquetas = torch.from_numpy(etiq).long()

        modelo.eval() # Desactiva los gradientes

        with torch.no_grad():
            outputs = modelo(datos)
        
        _, predicciones = torch.max(outputs, 1)
        
        precision = (predicciones == etiquetas).sum().item() / len(etiquetas)
        print('La precision del modelo es del ' + str(round(precision * 100, 5)) + '%.')

        return precision, predicciones
    
def registrar(precision, error, carac, fecha, hito):
    ruta = os.path.dirname(os.getcwd()) + '\ExtracurricularSinc\historial.csv'
    if os.path.exists(ruta):
        with open(ruta, mode='a', newline='') as archivo:
            escritor_csv = csv.writer(archivo)
            escritor_csv.writerow([precision, error, carac, fecha, hito])
    else:
        with open(ruta, mode='w', newline='') as archivo:
            escritor_csv = csv.writer(archivo)
            escritor_csv.writerow([precision, error, carac, fecha, hito])
    print('Se registra:')
    print(precision, error, carac, fecha, hito)

def grafico_oscuro(ejex = 'x', ejey = 'y', titulo = '', ax = plt.gca()):
    ax = plt.gca()
    ax.set_xlabel(ejex, fontsize=12, color = 'white')
    ax.set_ylabel(ejey, fontsize=12, color = 'white')
    ax.set_title(titulo, fontsize=24, color = 'white')
    plt.style.use('dark_background')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.legend(loc='best', bbox_to_anchor=(1, 1), fontsize=8, facecolor='black', edgecolor='white', labelcolor='white')
    plt.tight_layout()

