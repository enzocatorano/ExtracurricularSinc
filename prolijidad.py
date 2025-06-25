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

def prueba_prediccion_estimulo ():

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

####################################################################################################################################
####################################################################################################################################
# PRUEBO LA CONVERSION DE DATOS A STFT
# es para ver que la funcion funcione bien
####################################################################################################################################

def prueba_conversion_stft():

    import buenas_practicas as mbp
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from torch.utils.data import DataLoader
    from torch import nn, optim

    sujeto = 1
    prediccion = 'estimulo'
    datos = mbp.DataSetEEG(sujeto)
    f_max = 64
    t_ventana = 1024
    n = 256
    t_solapamiento = t_ventana*(n-1)/n
    fs = 1024
    datos.convertir_a_stft(f_max = f_max, t_ventana = t_ventana, t_solapamiento = t_solapamiento, fs = fs, aplanado = False)
    t = datos.tiempo_stft
    f = datos.frecuencia_stft

    # grafico el stft
    azar = np.random.randint(0, len(datos))
    print(datos[azar][0].shape)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(3, 2, figsize=(18, 8))
    for i in range(2):
        for j in range(3):
            idx = i*3 + j
            dato_plot = datos[azar][0][idx]
            # quiero pasar los datos de la matriz por una sigmoide
            dato_plot = np.sqrt(dato_plot)
            ax[idx // 2, idx % 2].pcolormesh(t, f, dato_plot, shading='nearest', cmap='cividis')
            ax[idx // 2, idx % 2].set_title(f'Canal {idx + 1}')
            ax[idx // 2, idx % 2].set_ylabel('Frecuencia [Hz]')
            ax[idx // 2, idx % 2].set_xlabel('Tiempo [s]')
    plt.tight_layout()
    plt.show()


    # datos.dejar_etiqueta(prediccion)
    # datos_vocales, datos_comandos = datos.split_estimulo_datasets()

    # entrenamiento, validacion, prueba = datos.particionar(0.7, True)
    # datos.normalizar()

####################################################################################################################################
####################################################################################################################################
# CARGA DE UN MODELO AE ENTRENADO
# intento ver si puedo cargar el modelo guardado en "model.pmt" para hacerle pruebas
####################################################################################################################################

def cargar_AE_entrenado ():

    import buenas_practicas as mbp
    import ClaseAE as cae
    import torch as tr
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from sklearn.manifold import TSNE
    
    # cargo el modelo
    # Hyperparámetros (deben coincidir con los usados en el entrenamiento)
    sujeto = 1
    prediccion = 'estimulo'
    datos = mbp.DataSetEEG(sujeto)
    datos.dejar_etiqueta(prediccion)
    entrenamiento, validacion, prueba = datos.particionar(0.7, True, semilla = 123)
    
    cargador_entrenamiento = DataLoader(entrenamiento, batch_size=1, shuffle=False)
    cargador_validacion = DataLoader(validacion, batch_size=1, shuffle=False)
    cargador_prueba = DataLoader(prueba, batch_size=1, shuffle=False)

    arq_encoder = [entrenamiento[0][0].shape[0]]
    z_dim = 1024
    arq_decoder = [entrenamiento[0][0].shape[0]]
    func_act = 'relu'
    func_act_ultima_capa = False
    usar_batch_norm = True
    dropout = 0.05
    lr = 5e-4
    metodo_init_pesos = tr.nn.init.xavier_normal_
    DEVICE = 'cuda' if tr.cuda.is_available() else 'cpu'

    # Instanciar el modelo con la misma arquitectura
    modelo_cargado = cae.AE(arq_encoder=arq_encoder,
                 z_dim=z_dim,
                 arq_decoder=arq_decoder,
                 func_act=func_act,
                 func_act_ultima_capa=func_act_ultima_capa,
                 usar_batch_norm=usar_batch_norm,
                 dropout=dropout,
                 device=DEVICE,
                 log_dir='runs/ae_remastered', # No es estrictamente necesario para cargar, pero es bueno mantenerlo
                 lr = lr,
                 metodo_init_pesos = metodo_init_pesos)

    # Cargar los pesos guardados
    try:
        modelo_cargado.load_state_dict(tr.load('ae_sujeto1_dr_bn_5e-4', map_location=DEVICE))
        print("Modelo cargado exitosamente.")
    except FileNotFoundError:
        print(f"Error: El archivo del modelo 'ae_sujeto1_dr_bn_5e-4' no se encontró.")
        return
    
    modelo_cargado.eval() # Poner el modelo en modo evaluación

    print('#####################################################################')
    # en todo el dataset de prueba y validacion (juntos)
    # calculo la similaridad cosen y SNR promedio
    modelo_cargado.to(DEVICE)
    similarity = []
    snr = []
    for x, _ in cargador_prueba:
        with tr.no_grad():
            x = x.to(DEVICE)
            xhat = modelo_cargado(x)
            similarity.append(tr.cosine_similarity(x, xhat, dim=1))
            snr.append(10 * tr.log10(tr.sum(x ** 2) / tr.sum((x - xhat) ** 2)))
    similarity = tr.cat(similarity)
    snr = tr.stack(snr)
    print('En los datos de prueba')
    print(f'Similaridad coseno promedio: {similarity.mean().item()}')
    print(f'SNR promedio: {snr.mean().item()}')
    print('#####################################################################')

    print('Pruebo dato al azar')
    modelo_cargado.to(DEVICE)
    dato_azar_cpu = prueba[np.random.randint(0, len(prueba))][0] # Shape: [x_size]
    # Add a batch dimension and move to the correct device
    dato_azar_batched_cpu = dato_azar_cpu.unsqueeze(0) # Shape: [1, x_size]
    dato_azar_on_device = dato_azar_batched_cpu.to(DEVICE)

    z = modelo_cargado.encode(dato_azar_on_device)
    xhat_on_device = modelo_cargado.decode(z)

    # grafico la señal original y reconstruida en la parte superior de una figura de 2 graficos
    # en el inferior muestro la fft de la original y reconstruida tambien
    tiempo = np.arange(0, dato_azar_on_device.shape[1]) / 1024
    plt.style.use('dark_background')
    plt.figure(figsize = (18, 6))
    plt.subplot(2, 1, 1)
    plt.plot(tiempo, dato_azar_on_device.squeeze().cpu().detach().numpy(), color='dodgerblue', label='Original')
    plt.plot(tiempo, xhat_on_device.squeeze().cpu().detach().numpy(), color='orange', label='Reconstruida')
    plt.title('Señal Original vs. Reconstruida')
    plt.xlabel('Muestras')
    plt.ylabel('Amplitud')
    plt.grid(linewidth=0.2)
    plt.legend()

    plt.subplot(2, 1, 2)
    # Calcular FFT
    fft_original = tr.fft.fft(dato_azar_on_device.squeeze().cpu().detach())
    fft_reconstruida = tr.fft.fft(xhat_on_device.squeeze().cpu().detach())

    # Obtener magnitudes y frecuencias
    magnitudes_original = tr.abs(fft_original)
    magnitudes_reconstruida = tr.abs(fft_reconstruida)
    
    # Solo la mitad positiva del espectro
    n = len(dato_azar_on_device.squeeze())
    frecuencias = tr.fft.fftfreq(n, 1/1024) # Asumiendo frecuencia de muestreo de 1

    # quiero graficar solo hasta 64Hz
    frecuencias = frecuencias[:n//2]
    magnitudes_original = magnitudes_original[:n//2]
    magnitudes_reconstruida = magnitudes_reconstruida[:n//2]
    # Filtrar hasta 64 Hz
    idx_64hz = (tr.abs(frecuencias - 64)).argmin() + 1 # Encuentra el índice más cercano a 64 Hz
    
    plt.plot(frecuencias[:idx_64hz], magnitudes_original[:idx_64hz], color='dodgerblue', label='FFT Original')
    plt.plot(frecuencias[:idx_64hz], magnitudes_reconstruida[:idx_64hz], color='orange', label='FFT Reconstruida')
    plt.title('Espectro de Frecuencia (FFT)')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud')
    plt.grid(linewidth=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

    #################################################################################
    # 4. Quiero pasar todo el dataset de prueba por un tSNE 2D y verlo

    print("\n--- Visualizando el espacio latente con t-SNE ---")
    DEVICE = 'cuda' if tr.cuda.is_available() else 'cpu'

    # Obtener todas las muestras
    all_data = []
    all_labels = []
    for x, y in cargador_entrenamiento:
        all_data.append(x)
        all_labels.append(y)
    for x, y in cargador_validacion:
        all_data.append(x)
        all_labels.append(y)
    for x, y in cargador_prueba:
        all_data.append(x)
        all_labels.append(y)

    all_data_tensor = tr.cat(all_data).to(DEVICE)
    all_labels_tensor = tr.cat(all_labels)

    # Codificar los datos para obtener las representaciones latentes (z)
    with tr.no_grad():
        latent_representations = modelo_cargado.encode(all_data_tensor).cpu().numpy()

    # Aplicar t-SNE
    print(f"Aplicando t-SNE a {latent_representations.shape[0]} muestras...")
    tsne = TSNE(n_components = 2, perplexity = 20, max_iter = 3000)
    tsne_results = tsne.fit_transform(latent_representations)

    # quiero hacer 3 graficos, uno al lado del otro
    # siendo cada uno una representacion tSNE 2D de los datos
    # coloreando cada uno de los 3 tipos de etiqueta

    # Graficar los resultados de t-SNE
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # 1 fila, 3 columnas
    # Etiquetas: modalidad, estimulo, artefacto
    labels_to_plot = ['modalidad', 'estimulo', 'artefacto']
    titles = ['Modalidad', 'Estímulo', 'Artefacto']
    label_indices = [0, 1, 2] # Corresponden a las columnas en y_completa

    for i, (label_type, title, idx) in enumerate(zip(labels_to_plot, titles, label_indices)):
        # Obtener las etiquetas originales del dataset completo
        # Asegurarse de que las etiquetas estén en el mismo orden que all_data_tensor
        original_labels = datos.y_completa[:, idx]

        # Ajustar las etiquetas para que sean 0-indexadas si es necesario
        if original_labels.min() == 1:
            original_labels = original_labels - 1

        # Mapear etiquetas numéricas a nombres descriptivos para la leyenda
        if label_type == 'modalidad':
            class_names = ['Imaginada', 'Pronunciada']
        elif label_type == 'estimulo':
            class_names = ['A', 'E', 'I', 'O', 'U', 'Arriba', 'Abajo', 'Izquierda', 'Derecha', 'Adelante', 'Atras']
        elif label_type == 'artefacto':
            class_names = ['Limpio', 'Parpadeo']
        else:
            class_names = [str(j) for j in np.unique(original_labels)]

        scatter = axes[i].scatter(tsne_results[:, 0], tsne_results[:, 1], c=original_labels, cmap = ('gist_rainbow' if label_type == 'estimulo' else 'viridis'), alpha=0.7)
        axes[i].set_title(f't-SNE del Espacio Latente por {title}')
        axes[i].set_xlabel('Componente t-SNE 1')
        axes[i].set_ylabel('Componente t-SNE 2')
        
        # Crear leyenda personalizada
        handles, _ = scatter.legend_elements()
        legend_labels = [class_names[int(val)] for val in np.unique(original_labels)]
        axes[i].legend(handles,class_names, title=f'Clases de {title}')

    plt.tight_layout()
    plt.show()

####################################################################################################################################
####################################################################################################################################
# AE + MLP para predecir
# uso el embbeding obtenido por el AE como entrada del MLP clasificador
####################################################################################################################################
def AE_MLP_clasificador ():

    import buenas_practicas as mbp
    import ClaseAE as cae
    import torch as tr
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from torch import nn, optim

    # Cargar el modelo AE entrenado
    sujeto = 1
    prediccion = 'estimulo' # La predicción del MLP será 'estimulo'
    datos = mbp.DataSetEEG(sujeto)
    datos.dejar_etiqueta(prediccion) # El dataset principal se prepara para la etiqueta del MLP

    # Particionar los datos para el entrenamiento del MLP
    entrenamiento_ae, validacion_ae, prueba_ae = datos.particionar(0.7, True, semilla = 456)

    # Hyperparámetros del AE (deben coincidir con los usados en el entrenamiento del AE)
    arq_encoder = [entrenamiento_ae[0][0].shape[0]]
    z_dim = 1024
    arq_decoder = [entrenamiento_ae[0][0].shape[0]]
    func_act_ae = 'relu'
    func_act_ultima_capa_ae = False
    usar_batch_norm_ae = True
    dropout_ae = 0.05
    lr_ae = 5e-4
    metodo_init_pesos_ae = tr.nn.init.xavier_normal_
    DEVICE = 'cuda' if tr.cuda.is_available() else 'cpu'
    # Instanciar el modelo AE
    modelo_ae = cae.AE(arq_encoder=arq_encoder,
                 z_dim=z_dim,
                 arq_decoder=arq_decoder,
                 func_act=func_act_ae,
                 func_act_ultima_capa=func_act_ultima_capa_ae,
                 usar_batch_norm=usar_batch_norm_ae,
                 dropout=dropout_ae,
                 device=DEVICE,
                 log_dir='runs/ae_remastered',
                 lr = lr_ae,
                 metodo_init_pesos = metodo_init_pesos_ae)
    # Cargar los pesos guardados del AE
    try:
        modelo_ae.load_state_dict(tr.load('ae_sujeto1_dr_bn_5e-4', map_location=DEVICE))
        print("Modelo AE cargado exitosamente.")
    except:
        print(f"Error: El archivo del modelo 'ae_sujeto1_dr+bn' no se encontró.")
        return
    modelo_ae.eval() # Poner el modelo en modo evaluación
    # Congelar los pesos del encoder del AE
    for param in modelo_ae.encoder.parameters():
        param.requires_grad = False

    # creo el mlp
    modelo_mlp = mbp.MLP(arq = [z_dim, 128, 64, 11],
                         func_act = 'relu',
                         usar_batch_norm = False,
                         dropout = 0.05,
                         metodo_init_pesos = tr.nn.init.xavier_uniform_)
    
    # defino al entrenador
    entrenador = mbp.Entrenador(
                                modelo = modelo_mlp,
                                optimizador = optim.Adam(modelo_mlp.parameters(), lr = 1e-5, weight_decay = 1e-7),
                                func_perdida = nn.CrossEntropyLoss(),
                                device = 'cuda' if tr.cuda.is_available() else 'cpu',
                                parada_temprana = 50,
                                log_dir = 'runs/AE_MLP_clasificador'
                                )
    
    def crear_dataset_latente(modelo_ae, subset_original, device, batch_size=32):
        """
        Pasa un dataset a través del encoder de un AE y devuelve un TensorDataset
        con las representaciones latentes y las etiquetas originales.
        Este enfoque procesa los datos en lotes, lo que es mucho más eficiente.
        """
        modelo_ae.eval()
        # Usamos un DataLoader para procesar en lotes, es mucho más rápido
        cargador = DataLoader(subset_original, batch_size=batch_size, shuffle=False)
        
        todas_latentes = []
        todas_etiquetas = []
        
        with tr.no_grad():
            for x_batch, y_batch in cargador:
                x_batch = x_batch.to(device)
                z_batch = modelo_ae.encode(x_batch)
                todas_latentes.append(z_batch.cpu())
                todas_etiquetas.append(y_batch)
                
        tensor_latentes = tr.cat(todas_latentes, dim=0)
        tensor_etiquetas = tr.cat(todas_etiquetas, dim=0)
        
        return tr.utils.data.TensorDataset(tensor_latentes, tensor_etiquetas)

    # Hago pasar todo el dataset por el AE para crear los datos para alimentar al MLP
    print("Creando datasets con representaciones latentes para el MLP...")
    batch_size_ae_proc = 32 # batch size para la codificación
    entrenamiento_mlp = crear_dataset_latente(modelo_ae, entrenamiento_ae, DEVICE, batch_size_ae_proc)
    validacion_mlp = crear_dataset_latente(modelo_ae, validacion_ae, DEVICE, batch_size_ae_proc)
    prueba_mlp = crear_dataset_latente(modelo_ae, prueba_ae, DEVICE, batch_size_ae_proc)
    # ahora tengo que normalizar todos los datasets segun los de entrenamiento  usando normalizacion min-max
    all_train_latents = entrenamiento_mlp.tensors[0]
    #print(all_train_latents.shape)
    min_val = all_train_latents.min()
    max_val = all_train_latents.max()
    # Función de normalización
    def normalizar_tensor_dataset(dataset, min_val, max_val):
        normalized_latents = (dataset.tensors[0] - min_val) / (max_val - min_val)
        return tr.utils.data.TensorDataset(normalized_latents, dataset.tensors[1])
    entrenamiento_mlp = normalizar_tensor_dataset(entrenamiento_mlp, min_val, max_val)
    validacion_mlp = normalizar_tensor_dataset(validacion_mlp, min_val, max_val)
    prueba_mlp = normalizar_tensor_dataset(prueba_mlp, min_val, max_val)
    print("Datasets latentes creados.")

    # crear DataLoaders para el MLP
    batch_size = 51
    cargador_entrenamiento_mlp = DataLoader(entrenamiento_mlp, batch_size = batch_size, shuffle = True, drop_last = False)
    cargador_validacion_mlp = DataLoader(validacion_mlp, batch_size = batch_size, shuffle = False)
    cargador_prueba_mlp = DataLoader(prueba_mlp, batch_size = 1, shuffle = False)

    # Entrenar el MLP con los datos latentes
    print("\nEntrenando el clasificador MLP con el espacio latente...")
    entrenador.ajustar(
        cargador_entrenamiento = cargador_entrenamiento_mlp,
        cargador_validacion = cargador_validacion_mlp,
        epocas = 1000
    )
    # Evaluar el MLP
    print("\nEvaluando el clasificador MLP...")
    evaluador = mbp.Evaluador(modelo = modelo_mlp, device = DEVICE, clases='estimulo')
    evaluador.matriz_confusion(cargador_prueba_mlp)


####################################################################################################################################
####################################################################################################################################
# correr

if __name__ == '__main__':
    #prueba_conversion_stft()
    #cargar_AE_entrenado()
    #prueba_prediccion_estimulo()
    AE_MLP_clasificador()


####################################################################################################################################