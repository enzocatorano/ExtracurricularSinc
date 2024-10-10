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

# Cargo el dataset
import scipy.io
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

def extraer_eeg (ruta_entrada):
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
            datos = np.array([F3, F4, C3, C4, P3, P4]).T
            etiquetas = np.array([modalidad, estimulo, artefacto])
            np.save(ruta_datos, datos)
            np.save(ruta_etiquetas, etiquetas)
        else:
            print('Los datos de este sujeto ya fueron extraidos, no se reescribiran.')
        print('Datos del sujeto extraidos correctamente.')
    print('Extraccion total finalizada')
    return

def graficar (sujeto, ruta, mod = 0):
    sujeto = str(sujeto)
    if len(sujeto) == 1:
        sujeto = '0' + sujeto
    ruta_datos = 'Datos/' + ruta + '/' + sujeto + '_datos.npy'
    ruta_etiquetas = 'Datos/' + ruta + '/' + sujeto + '_etiquetas.npy'
    datos = np.load(ruta_datos)
    etiquetas = np.load(ruta_etiquetas)
    print(datos)
    print(type(datos), datos.shape)
    F3 = datos[:,0]
    F4 = datos[:,1]
    C3 = datos[:,2]
    C4 = datos[:,3]
    P3 = datos[:,4]
    P4 = datos[:,5]
    modalidad = etiquetas[0]
    estimulo = etiquetas[1]
    artefacto = etiquetas[2]
    n = 4096
    modalidad_posible = ['Imaginada', 'Pronunciada']
    estimulo_posible = ['A', 'E', 'I', 'O', 'U', 'Arriba', 'Abajo',
                  'Adelante', 'Atras', 'Derecha', 'Izquierda']
    artefacto_posible = ['Ninguno', 'Parpadeo']
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
            print(estimulo[i])
            plt.axvspan(i*n, n*(i + 1), alpha = 0.3, color = colores[pos])
            i += 1
    plt.legend()
    plt.show()

# extraer_eeg('imagined_speech')
graficar(1, 'Solo_ordenados', 1)

