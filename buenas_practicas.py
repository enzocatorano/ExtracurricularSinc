# LAS BUENAS PRACTICAS

# Despues de algunas clases de deep learning, me di cuenta de que tengo que
# arreglar tantos detalles de las funciones que hice, que lo mejor va a
# ser que las rehaga de forma correcta.

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from tqdm import tqdm
import datetime
import os
import copy
import scipy.io as sio
import mne
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from scipy.signal import stft


############################################################################################################
############################################################################################################
############################################################################################################


class MLP (nn.Module):

    def __init__(self, arq : list, func_act = 'relu',
                 usar_batch_norm = True,
                 dropout : float = None,
                 metodo_init_pesos = None,
                 semilla = None):
        '''
        arq :
            Una lista de valores enteros que indican la cantidad de neuronas por
            capa. Debe tener al menos dos elementos, con las neuronas de entrada
            y salida. Como ejemplo:
                arq = [32, 80, 20, 5]
            indica que se tienen cuatro capas, con 32, 80, 20 y 5 neuronas,
            respectivamente. Donde 32 es la entrada, y 5 la salida.
        func_act :
            Admite 4 formatos, str 'relu', 'sigmoid', 'tanh' o 'leakyrelu',
            indicando el tipo de funcion de activacion a implementar en todas las
            neuronas y capas. O formato lista, de igual a la longitud de arq
            menos uno, de forma que se detallen las funciones de activacion de
            cada capa. Por ejemplo:
                func_act = 'relu'
                func_act = ['sigmoid', 'relu', 'relu']
        usar_batch_norm :
            True/False para aplicar BatchNorm1d en capas ocultas.
        dropout :
            Float en [0,1] para probabilidad de Dropout en capas ocultas.
        metodo_init_pesos :
            Función de inicialización (ej. nn.init.xavier_uniform_) o None.
        semilla :
            Entero para reproducibilidad (CPU y CUDA).
        '''
        super().__init__()

        # reproducibilidad
        if semilla is not None:
            torch.manual_seed(semilla)
            torch.cuda.manual_seed_all(semilla)
            self.semilla = semilla

        # preparar lista de activaciones y convertir a módulos
        if isinstance(func_act, str):
            func_list = [func_act] * (len(arq) - 1)
        elif isinstance(func_act, list) and len(func_act) == len(arq) - 1:
            func_list = func_act
        else:
            raise ValueError('func_act debe ser str o lista de longitud len(arq)-1')
        self.func_act = self._dar_activaciones(func_list)

        # validar dropout y batch normalization
        if dropout is not None and (dropout < 0 or dropout > 1):
            raise ValueError('dropout debe estar entre 0 y 1.')
        self.use_batch_norm = usar_batch_norm
        self.dropout_rate = dropout

        # construir bloques nombrados con OrderedDict
        bloques = OrderedDict()
        for i in range(len(arq) - 1):
            in_f, out_f = arq[i], arq[i+1]
            # capa lineal
            bloques[f"linear{i}"] = nn.Linear(in_f, out_f)
            # batchnorm solo en capas ocultas
            if usar_batch_norm and i < len(arq) - 2:
                bloques[f"batchnorm{i}"] = nn.BatchNorm1d(out_f)
            # activación
            bloques[f"act{i}"] = self.func_act[i]
            # dropout solo en capas ocultas
            if dropout > 0 and i < len(arq) - 2:
                bloques[f"dropout{i}"] = nn.Dropout(dropout)
        # secuencial con nombres legibles
        self.estructura_total = nn.Sequential(bloques)

        # inicializacion de pesos
        if metodo_init_pesos is not None:
            self._inicializar_pesos(metodo_init_pesos)


    #########################################################################
    # algunos metodos privados:
    def _dar_activaciones(self, func_list):
        '''
        Convierte lista de strings a módulos de activación.
        '''
        acts = []
        for a in func_list:
            a_low = a.lower()
            if a_low == 'relu':
                acts.append(nn.ReLU())
            elif a_low == 'sigmoid':
                acts.append(nn.Sigmoid())
            elif a_low == 'tanh':
                acts.append(nn.Tanh())
            elif a_low == 'leakyrelu':
                acts.append(nn.LeakyReLU())
            else:
                raise ValueError("Activación debe ser 'relu', 'sigmoid', 'tanh' o 'leakyrelu'.")
        return acts
    
    def _inicializar_pesos (self, metodo_init_pesos):
        '''
        Inicializa los pesos segun el metodo especificado.
        '''
        for mod in self.estructura_total:
            if isinstance(mod, nn.Linear):
                metodo_init_pesos(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
    #########################################################################

    def __str__(self):
        desc = []
        idx = 0
        n_layers = len(self.estructura_total)
        for module in self.estructura_total:
            desc.append(f"[{idx}] {module}")
            idx += 1
        return "\n".join(desc)


    def forward(self, x):
        return self.estructura_total(x)


############################################################################################################
############################################################################################################
############################################################################################################


class Entrenador ():
    '''
    Clase para entrenar y evaluar modelos PyTorch de forma genérica,
    con registro de métricas y pesos en TensorBoard.
    '''
    def __init__(self, modelo: nn.Module,
                 optimizador: optim.Optimizer = None,
                 func_perdida: nn.Module = None,
                 device: torch.device = None,
                 parada_temprana: int = None,
                 log_dir: str = 'runs'):

        # dispositivo en el que se correra el entrenamiento
        if device is not None and device == 'cuda':
            if torch.cuda.is_available():
                self.device = device
                print('Utilizando GPU.')
            else:
                self.device = 'cpu'
                print('No se encontro GPU, se utilizara CPU.')
        else:
            self.device = 'cpu'
            print('Se utilizara CPU.')

        if optimizador is None:
            optimizador = optim.Adam(modelo.parameters(), lr = 1e-3)
        if func_perdida is None:
            raise ValueError('Falto especificar funcion perdida.')

        self.modelo = modelo.to(self.device)
        self.optimizador = optimizador
        self.func_perdida = func_perdida
        self.parada_temprana = parada_temprana

        # Guardar el directorio base de logs proporcionado por el usuario
        self.base_log_dir = log_dir
        # Crear un subdirectorio único para esta ejecución específica usando un timestamp
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        # run_specific_log_dir será algo como 'runs/mi_experimento/20230101-120000'
        self.run_specific_log_dir = os.path.join(self.base_log_dir, current_time)
        # SummaryWriter creará este directorio si no existe.
        self.escritor = SummaryWriter(log_dir=self.run_specific_log_dir)


    def _epoca_entrenamiento (self, cargador_entrenamiento: DataLoader, epoca: int):
        '''
        Ejecuta una época de entrenamiento.
        '''
        self.modelo.train()
        perdida_total = 0.0
        for x, y in tqdm(cargador_entrenamiento,
                         desc=f"Epoca {epoca} Entrenamiento",
                         leave = False):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizador.zero_grad()
            predicciones = self.modelo(x)
            perdida = self.func_perdida(predicciones, y)
            perdida.backward()
            self.optimizador.step()
            perdida_total += perdida.item()*x.shape[0]
        perdida_promedio_epoca = perdida_total / len(cargador_entrenamiento.dataset)
        self.escritor.add_scalar('Perdida/entrenamiento', perdida_promedio_epoca, epoca)
        return perdida_promedio_epoca


    def _epoca_validacion (self, cargador_validacion: DataLoader, epoca: int):
        '''
        Ejecuta una época de validación.
        '''
        self.modelo.eval()
        perdida_total = 0.0
        correctas = 0
        with torch.no_grad():
            for x, y in tqdm(cargador_validacion,
                         desc=f"Epoca {epoca} ✅ Validación",
                         leave = False):
                x, y = x.to(self.device), y.to(self.device)
                predicciones = self.modelo(x)
                perdida = self.func_perdida(predicciones, y)
                perdida_total += perdida.item()*x.shape[0]
                if predicciones.dim() > 1:
                    _, predicciones = torch.max(predicciones, 1)
                    correctas += (predicciones == y).sum().item()
            perdida_promedio_epoca = perdida_total / len(cargador_validacion.dataset)
            precision = correctas / len(cargador_validacion.dataset)
            self.escritor.add_scalar('Perdida/validacion', perdida_promedio_epoca, epoca)
            self.escritor.add_scalar('Precision/validacion', precision, epoca)
        return perdida_promedio_epoca, precision


    def ajustar (self,
                 cargador_entrenamiento: DataLoader,
                 cargador_validacion: DataLoader = None,
                 epocas: int = 100,
                 nombre_modelo_salida: str = None):
        '''
        Ajusta el modelo.
        Verifica compatibilidad entre la salida del modelo, las etiquetas y la funcion perdida.

        nombre_modelo_salida:
            Si se proporciona, guarda el mejor modelo en esta ruta.
        '''
        x, y = next(iter(cargador_entrenamiento))
        self._revisar_formato_etiquetas(self.modelo(x.to(self.device)), y)

        mejor_perdida_validacion = float('inf')
        epocas_sin_mejora = 0
        mejor_modelo = copy.deepcopy(self.modelo.state_dict())


        for epoca in tqdm(range(1, epocas + 1),
                          desc = "Entrenamiento total",
                          unit = "época"):
            perdida_entrenamiento = self._epoca_entrenamiento(cargador_entrenamiento, epoca)
            if cargador_validacion is not None:
                perdida_validacion, precision = self._epoca_validacion(cargador_validacion, epoca)
                if perdida_validacion < mejor_perdida_validacion:
                    mejor_perdida_validacion = perdida_validacion
                    epocas_sin_mejora = 0
                    mejor_modelo = copy.deepcopy(self.modelo.state_dict())
                    if nombre_modelo_salida:
                        torch.save(self.modelo.state_dict(), nombre_modelo_salida)
                else:
                    epocas_sin_mejora += 1
                if self.parada_temprana is not None and epocas_sin_mejora >= self.parada_temprana:
                    print(f'Se ejecuta la parada temprana tras {epoca} épocas.')
                    break
            self._log_pesos_y_gradientes(epoca)
        if cargador_validacion is not None:
            self.modelo.load_state_dict(mejor_modelo)
        self.escritor.close()
        print(f"\nEntrenamiento completado.")
        print(f"Logs guardados en: {self.run_specific_log_dir}")
        print(f"Para ver este run específico en TensorBoard: tensorboard --logdir=\"{self.run_specific_log_dir}\"")
        print(f"Para comparar runs dentro del experimento ({self.base_log_dir}): tensorboard --logdir=\"{self.base_log_dir}\"")


    def _log_pesos_y_gradientes (self, epoca: int):
        '''
        Registra pesos y gradientes en TensorBoard.
        '''
        for nombre, parametro in self.modelo.named_parameters():
            self.escritor.add_histogram(f'Parametros/{nombre}', parametro.data.detach().cpu(), epoca)
            if parametro.grad is not None:
                self.escritor.add_histogram(f'Gradientes/{nombre}', parametro.grad.detach().cpu(), epoca)


    def _revisar_formato_etiquetas (self, preds: torch.Tensor, etiquetas: torch.Tensor):
        '''
        Verifica compatibilidad entre etiquetas y funcion perdida.
        '''
        if isinstance(self.func_perdida, nn.CrossEntropyLoss):
            if etiquetas.dim() > 1:
                raise ValueError(f'Las etiquetas deben tener formato [batch_size,].')
        elif isinstance(self.func_perdida, nn.MSELoss):
            if etiquetas.shape != preds.shape:
                raise ValueError(f'Los tensores de etiquetas y predicciones deben tener el mismo formato.')

# falta:
#   la posibilidad de usar regularizacion


############################################################################################################
############################################################################################################
############################################################################################################


class Evaluador ():
    '''
    Clase para evaluar modelos PyTorch de forma genérica.
    Requiere que se haya usado la funcion dejar_etiqueta().
    '''
    def __init__ (self, modelo : nn.Module, device : torch.device = None, clases: str = None):

        if device is None and device == 'cuda':
            if torch.cuda.is_available():
                self.device = device
                print('Utilizando GPU.')
            else:
                self.device = 'cpu'
                print('No se encontro GPU, se utilizara CPU.')
        else:
            self.device = 'cpu'
            print('Se utilizara CPU.')

        self.modelo = modelo.to(self.device)
        self.clases = clases
        self.nombres_clases = None
        if self.clases:
            self._set_nombres_clases()


    def _set_nombres_clases(self):
        """Define los nombres de las clases para los gráficos basado en self.clases."""
        if self.clases == 'modalidad':
            self.nombres_clases = ['Imaginada', 'Pronunciada']
        elif self.clases == 'estimulo':
            self.nombres_clases = ['A', 'E', 'I', 'O', 'U', 'Arriba', 'Abajo', 'Izquierda', 'Derecha', 'Adelante', 'Atras']
        elif self.clases == 'artefacto':
            self.nombres_clases = ['Limpio', 'Parpadeo']
        elif self.clases == 'estimulo_binario':
            self.nombres_clases = ['Vocal', 'Comando']
        elif self.clases == 'estimulo_vocal':
            self.nombres_clases = ['A', 'E', 'I', 'O', 'U']
        elif self.clases == 'estimulo_comando':
            self.nombres_clases = ['Arriba', 'Abajo', 'Izquierda', 'Derecha', 'Adelante', 'Atras']


    def _inferir_clases(self, cargador):
        """Intenta inferir las clases del dataset si no se proveyeron."""
        if self.clases: # Si ya se proveyeron, no hacer nada.
            return

        dataset_obj = cargador.dataset
        while isinstance(dataset_obj, Subset):
            dataset_obj = dataset_obj.dataset

        if hasattr(dataset_obj, 'etiqueta'):
            self.clases = dataset_obj.etiqueta
            self._set_nombres_clases()


    def probar (self, dataloader):
        self.modelo.eval()
        todas_preds = []
        todas_verdaderas = []

        # Intenta inferir las clases si no se pasaron en el constructor
        self._inferir_clases(dataloader)

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                salida = self.modelo(x)

                if salida.dim() > 1 and salida.shape[1] > 1:
                    pred = torch.argmax(salida, dim=1)
                    verdadera = torch.argmax(y, dim=1) if y.dim() > 1 else y
                else:
                    pred = (salida > 0.5).long().squeeze()
                    verdadera = y

                todas_preds.append(pred.cpu().numpy())
                todas_verdaderas.append(verdadera.cpu().numpy())

        return np.concatenate(todas_verdaderas), np.concatenate(todas_preds)


    def matriz_confusion(self, dataloader, plot = True, titulo = "Matriz de Confusión"):
        verdaderas, preds = self.probar(dataloader)
        cm = confusion_matrix(verdaderas, preds)
        if plot:
            plt.style.use('dark_background')
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="cividis",
                        xticklabels=self.nombres_clases if self.nombres_clases else np.unique(verdaderas),
                        yticklabels=self.nombres_clases if self.nombres_clases else np.unique(verdaderas))
            plt.xlabel("Etiqueta predicha")
            plt.ylabel("Etiqueta verdadera")
            plt.title(titulo)
            plt.tight_layout()
            plt.show()
        return cm
    

    def reporte(self, dataloader):
        verdaderas, preds = self.probar(dataloader)
        print(classification_report(
            verdaderas, preds,
            target_names = self.nombres_clases if self.nombres_clases else None,
            digits=3
        ))

############################################################################################################
############################################################################################################
############################################################################################################


class Gestor ():
    '''
    
    '''


############################################################################################################
############################################################################################################
############################################################################################################


class DataSetEEG (Dataset):
    '''
    Carga los datos de EEG, y deja a punto para usar en los modelos Pytorch.
    Si sujeto es None, se usa un objeto vacío; útil para crear datasets desde arrays.
    '''
    def __init__ (self, sujeto = None, ruta_sujetos = '../Base_de_Datos_Habla_Imaginada/'):
        '''
        Carga los datos de un sujeto.
        '''
        if sujeto is not None:
            self.sujeto = sujeto
            ruta = ruta_sujetos
            if sujeto < 10:
                ruta += f'S0{sujeto}/S0{sujeto}_EEG.mat'
            else:
                ruta += f'S{sujeto}/S{sujeto}_EEG.mat'
            
            # checkeo y carga
            try:
                data = sio.loadmat(ruta, squeeze_me = True, struct_as_record = False)
            except FileNotFoundError:
                raise FileNotFoundError(f'No se encontro el archivo en la ruta: {ruta}')
            EEG = data['EEG']

            # separando datos de etiquetas
            self.x = EEG[:, :-3]
            self.y_completa = EEG[:, -3:].copy()
            self.y = self.y_completa.copy()
        else:
            # para cuando se cree por arrays
            pass
        

    def __len__ (self):
        return self.x.shape[0]
    

    def __getitem__ (self, idx):
        current_x = self.x[idx] # Esto es un array de NumPy (características de una muestra)
        current_y_val = self.y[idx] # Esto puede ser un escalar de NumPy si self.y es 1D

        # Convertir a tensores de PyTorch
        # Es buena práctica asegurar el tipo de dato, ej: float para datos, long para etiquetas
        if isinstance(current_x, np.ndarray):
            x_tensor = torch.from_numpy(current_x).float()
        else: # Asumimos que ya es un torch.Tensor
            x_tensor = current_x.float()

        # Manejar y: si es un escalar (como numpy.int64), convertirlo primero a un array NumPy 0-D
        if not isinstance(current_y_val, np.ndarray):
            # Esto maneja el caso donde current_y_val es un escalar de NumPy (ej. np.int64)
            current_y_np_array = np.array(current_y_val)
        else:
            # Esto maneja el caso donde self.y sigue siendo 2D (ej. antes de llamar a dejar_etiqueta)
            current_y_np_array = current_y_val
        
        y_tensor = torch.from_numpy(current_y_np_array).long() # Asumiendo que las etiquetas deben ser long
        return x_tensor, y_tensor


    def graficar_canales (self, dato):
        x_selec = self.x[dato]
        y_selec = self.y[dato]
        t = np.arange(0, 4, 1/1024)
        etiquetas = [['Imaginada','Pronunciada'],
                     ['A', 'E', 'I', 'O', 'U', 'Arriba', 'Abajo', 'Adelante', 'Atras', 'Derecha', 'Izquierda'],
                     ['Limpio', 'Parpadeo']]
        canales = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4']
        etiq_dato = [etiquetas[0][int(y_selec[0])-1], etiquetas[1][int(y_selec[1])-1], etiquetas[2][int(y_selec[2])-1]]
        minimo, maximo = np.min(x_selec), np.max(x_selec)

        plt.style.use('dark_background')
        fig, axs = plt.subplots(6, 1, figsize = (12, 9))
        for i in range(6):
            axs[i].plot(t, x_selec[4096*i: 4096*(i+1)], color = 'dodgerblue')
            axs[i].set_ylabel('Amplitud')
            axs[i].grid(linewidth = 0.2, linestyle = '--')
            axs[i].set_xlim(-0.05, 4.15)
            axs[i].set_ylim(minimo*1.2 if minimo < 0 else minimo*0.8, maximo*1.2)
            axs[i].set_xticks(np.arange(0, 4.25, 0.25))
            if i != 5:
                axs[i].set_xticklabels([])
                # tambien quiero sacar las marquitas
                axs[i].tick_params(axis = 'x', which = 'both', bottom = False, top = False)
            # cuadro de texto con el canal
            axs[i].text(0.995, 0.18, f'{canales[i]}', transform = axs[i].transAxes, fontsize = 12,
                    verticalalignment = 'top', horizontalalignment = 'right',
                    bbox = dict(fc='black', ec='white'))
        # etiquetas arriba del primer subplot
        etiquetas_texto = f'{etiq_dato[0]} - {etiq_dato[1]} - {etiq_dato[2]}'
        axs[0].text(0.99, 1.05, etiquetas_texto, transform = axs[0].transAxes, fontsize = 10,
                    verticalalignment = 'bottom', horizontalalignment = 'right',
                    bbox=dict(fc='black', ec='white'))
        # ahora lo mismo con sujeto y dato
        axs[0].text(0.01, 1.05, f'Sujeto {self.sujeto} - Medicion {dato}', transform = axs[0].transAxes, fontsize = 10,
                    verticalalignment = 'bottom', horizontalalignment = 'left',
                    bbox=dict(fc='black', ec='white'))
        axs[-1].set_xlabel('Tiempo [s]')
        plt.show()
    

    @classmethod
    def desde_arrays (cls, x, y, etiqueta):
        '''
        Crea un DataSetEEG a partir de arrays numpy x e y (etiquetas).
        Útil para subdatasets filtrados.
        '''
        obj = cls(sujeto = None)
        obj.x = x
        obj.y_completa = None
        obj.y = y
        obj.etiqueta = etiqueta
        return obj

        
    def particionar (self, ratio_entrenamiento = 0.7, validacion = True, semilla = None):
        if ratio_entrenamiento >= 1 or ratio_entrenamiento <= 0:
            raise ValueError('El ratio de entrenamiento debe estar entre 0 y 1.')
        if semilla is None:
            semilla = np.random.randint(0, 9999999)
        indices = np.arange(len(self))
        indices_entrenamiento, indices_sobrantes = train_test_split(indices, train_size = ratio_entrenamiento, random_state = semilla)
        self.indices_entrenamiento = indices_entrenamiento
        subset_entrenamiento = Subset(self, indices_entrenamiento)
        if validacion:
            indices_validacion, indices_test = train_test_split(indices_sobrantes, train_size = 0.5, random_state = semilla)
            subset_validacion = Subset(self, indices_validacion)
            subset_prueba = Subset(self, indices_test)
            return subset_entrenamiento, subset_validacion, subset_prueba
        else:
            subset_prueba = Subset(self, indices_sobrantes)
            return subset_entrenamiento, subset_prueba


    def dejar_etiqueta (self, etiqueta : str):
        '''
        Mantiene una de las tres etiquetas del dataset
        (modalidad, estimulo o artefacto), para ser usada
        en prediccion.
        '''
        posibles = ['modalidad','estimulo','artefacto']
        if etiqueta not in posibles:
            raise ValueError(f'La etiqueta debe ser una de las siguientes: {posibles}')
        idx = posibles.index(etiqueta)
        self.y = self.y[:, idx]
        # Asegurarse de que las etiquetas sean 0-indexadas si es necesario para CrossEntropyLoss
        if self.y.min() == 1:
            self.y = self.y - 1
        self.etiqueta = etiqueta


    def binarizar_estimulo (self):
        '''
        Convierte la etiqueta de estimulo en binaria:
            vocal (0) y comando (1).
        '''
        if self.y_completa is None:
            raise ValueError('No hay etiquetas originales para binarizar.')
        est = self.y_completa[:, 1].astype(int)
        self.y = np.where(est <= 5, 0, 1).astype(int)
        self.etiqueta = 'estimulo_binario'


    def split_estimulo_datasets (self):
        '''
        Convierte primero la etiqueta de estímulo del dataset original a binaria,
        luego filtra por clase y devuelve dos DataSetEEG:
          - vocales (5 clases, etiquetas 0..4)
          - comandos (6 clases, etiquetas 0..5)
        '''
        self.binarizar_estimulo()

        est = self.y_completa[:, 1].astype(int)
        mascara_vocales = est <= 5
        mascara_comandos = est > 5
        x_vocales = self.x[mascara_vocales]
        y_vocales = est[mascara_vocales] - 1
        x_comandos = self.x[mascara_comandos]
        y_comandos = est[mascara_comandos] - 6

        dataset_vocales = DataSetEEG.desde_arrays(x_vocales, y_vocales, 'estimulo_vocal')
        dataset_comandos = DataSetEEG.desde_arrays(x_comandos, y_comandos, 'estimulo_comando')
        
        return dataset_vocales, dataset_comandos

    
    def normalizar (self):
        '''
        Aplica normalizacion min-max a todos los datos, segun
        los datos de entrenamiento.
        '''
        self.maximo = np.max(self.x[self.indices_entrenamiento])
        self.minimo = np.min(self.x[self.indices_entrenamiento])
        self.x = (self.x - self.minimo) / (self.maximo - self.minimo)


    def convertir_a_stft (self,
                          f_max,
                          t_ventana = 256,
                          t_solapamiento = 128,
                          fs = 1024,
                          aplanado = True):
        '''
        Reemplaza los datos originales por la STFT de los mismos.
        Solicita 4 parametros:
            f_max :
                Frecuencia maxima del espectrograma a retener. Todo lo que este
                por encima de esta, se perderá.
            t_ventana :
                Cantidad de muestras temporales de las ventanas de la STFT.
            t_solapamiento :
                Cantidad de muestras temporales de solapamiento entre ventanas.
            fs :
                Frecuencia de muestreo de los datos. Dejo la posibilidad de definirla
                por si se llegara a querer downsamplear.
        '''

        n_canales = 6
        puntos_por_canal = 4*fs

        stft_data = []
        for i in range(self.x.shape[0]): # Iterar sobre cada muestra
            sample_stft = []
            for j in range(n_canales): # Iterar sobre cada canal dentro de la muestra
                channel_data = self.x[i, j * puntos_por_canal : (j + 1) * puntos_por_canal]
                # Calcular STFT para el canal
                f, t, Zxx = stft(channel_data, fs=fs, nperseg = t_ventana, noverlap = t_solapamiento)

                # de Zxx me quedo solo con los valores absolutos de aquellos cuya frecuencia sea
                # menor o igual a f_max
                mascara_frecuencia = f <= f_max
                Zxx_filtrado = np.abs(Zxx[mascara_frecuencia, :])

                if aplanado:
                    # Aplanar el espectrograma filtrado y agregarlo a la lista de la muestra
                    sample_stft.append(Zxx_filtrado.flatten())
                else:
                    # Mantener la estructura 2D (frecuencia x tiempo)
                    sample_stft.append(Zxx_filtrado)

            if aplanado:
                # Concatenar los espectrogramas aplanados de todos los canales para la muestra
                stft_data.append(np.concatenate(sample_stft))
            else:
                # Apilar los espectrogramas 2D de todos los canales (nuevo eje para canales)
                # El resultado será (canales, frecuencias, tiempo)
                stft_data.append(np.stack(sample_stft, axis=0))

        ## Convertir la lista de arrays numpy a un único array numpy
        stft_data_np = np.array(stft_data)
        # Convierto el array numpy a tensor y lo guardo como self.x
        self.x = torch.tensor(stft_data_np, dtype=torch.float32)
        self.tiempo_stft = t
        mascara_frecuencia = f <= f_max
        self.frecuencia_stft = f[mascara_frecuencia]