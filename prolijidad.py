####################################################################################################################################
####################################################################################################################################
####################################################################################################################################

# ARCHIVO DE FORMALIDAD DEL TRABAJO
# un intento de que no sea un desastre lo que voy haciendo

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################




####################################################################################################################################
# PREDICCION DEL TIPO DE ESTIMULO
# quiero ver si puedo hacer que un MLP diferencia entre las vocales y los comandos
####################################################################################################################################

import buenas_practicas as mbp
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

sujeto = 1
prediccion = 'estimulo'
datos = mbp.DataSetEEG(sujeto)
datos.dejar_etiqueta(prediccion)
datos_vocales, datos_comandos = datos.split_estimulo_datasets()

entrenamiento, validacion, prueba = datos.particionar(0.7, True)
datos.normalizar()

arq = [entrenamiento[0][0].shape[0], 64, 64, 2]
func_act = 'relu'
modelo = mbp.MLP(arq,
                 func_act = func_act,
                 usar_batch_norm = True,
                 dropout = 0.05,
                 metodo_init_pesos = nn.init.xavier_uniform_)
entrenador = mbp.Entrenador(
                            modelo = modelo,
                            optimizador = optim.Adam(modelo.parameters(), lr = 1e-7, weight_decay = 1e-6),
                            func_perdida = nn.CrossEntropyLoss(),
                            device = 'cuda' if torch.cuda.is_available() else 'cpu',
                            parada_temprana = 100,
                            log_dir = 'runs/diferenciar_tipos_estimulo_mlp'
                            )
batch_size = 11
cargador_entrenamiento = DataLoader(entrenamiento, batch_size = batch_size, shuffle = True, drop_last = False)
cargador_validacion = DataLoader(validacion, batch_size = batch_size, shuffle = False)

entrenador.ajustar(cargador_entrenamiento, cargador_validacion, epocas = 5000)

cargador_prueba = DataLoader(prueba, batch_size = 1, shuffle = False)
evaluador = mbp.Evaluador(modelo = modelo,
                          device = 'cuda' if torch.cuda.is_available() else 'cpu')

evaluador.matriz_confusion(cargador_prueba)