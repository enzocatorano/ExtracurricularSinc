from torch import nn
import torch as tr
from tqdm import tqdm
import time
import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from collections import OrderedDict

class AE(nn.Module):
    def __init__(self,
                 arq_encoder: list,
                 z_dim: int,
                 arq_decoder: list,
                 func_act = 'sigmoid',
                 func_act_ultima_capa = False,
                 usar_batch_norm = True,
                 dropout : float = None,
                 device = 'cpu',
                 log_dir: str = 'runs/ae',
                 lr = 0.001,
                 metodo_init_pesos = None):
        super().__init__()
        
        ###########################################################################################
        # preparar lista de activaciones y convertir a módulos
        arq = arq_encoder + [z_dim] + arq_decoder
        if isinstance(func_act, str):
            func_list = [func_act] * (len(arq) - 1)
        elif isinstance(func_act, list) and len(func_act) == len(arq) - 1:
            func_list = func_act
        else:
            raise ValueError('func_act debe ser str o lista de longitud len(arq)-1')
        self.func_act = self._dar_activaciones(func_list)

        ###########################################################################################
        # validar dropout y batch normalization
        if dropout is not None and (dropout < 0 or dropout > 1):
            raise ValueError('dropout debe estar entre 0 y 1.')
        self.use_batch_norm = usar_batch_norm
        self.dropout_rate = dropout

        ###########################################################################################
        # revisar que la capa de entrada del encoder tenga igual dimension que salida del decoder
        if arq[0] != arq[-1]:
            raise ValueError('La entrada y salida del autoencoder deben tener la misma dimension.')

        ###########################################################################################
        # construir bloques encoder
        arq_encoder = arq_encoder + [z_dim]
        bloques_encoder = OrderedDict()
        for i in range(len(arq_encoder) - 1):
            in_f, out_f = arq_encoder[i], arq_encoder[i+1]
            # capa lineal
            bloques_encoder[f"linear{i}"] = nn.Linear(in_f, out_f)
            # batchnorm en todas las capas del encoder
            if usar_batch_norm:
                bloques_encoder[f"batchnorm{i}"] = nn.BatchNorm1d(out_f)
            # activación
            bloques_encoder[f"act{i}"] = self.func_act[i]
            # dropout
            if dropout > 0:
                bloques_encoder[f"dropout{i}"] = nn.Dropout(dropout)
        self.encoder = nn.Sequential(bloques_encoder)

        # construir bloques decoder
        arq_decoder = [z_dim] + arq_decoder
        bloques_decoder = OrderedDict()
        for i in range(len(arq_decoder) - 1):
            in_f, out_f = arq_decoder[i], arq_decoder[i+1]
            # capa lineal
            bloques_decoder[f"linear{i}"] = nn.Linear(in_f, out_f)
            # batchnorm en todas las capas menos la ultima
            # (por si no esta normalizada la entrada)
            if usar_batch_norm and i < len(arq_decoder) - 2:
                bloques_decoder[f"batchnorm{i}"] = nn.BatchNorm1d(out_f)
            # activación
            # revisa si la ultima capa debe llevar activacion
            if i == len(arq_decoder) - 2:
                if func_act_ultima_capa:
                    bloques_decoder[f"act{i}"] = self.func_act[i]
            else:
                bloques_decoder[f"act{i}"] = self.func_act[i]
            # dropout en todas las capas menos la ultima
            if dropout > 0 and i < len(arq_decoder) - 2:
                bloques_decoder[f"dropout{i}"] = nn.Dropout(dropout)
        self.decoder = nn.Sequential(bloques_decoder)
        
        ###########################################################################################
        # inicializacion de pesos segun metodo, llamando a un metodo privado
        if metodo_init_pesos is not None:
            self._inicializar_pesos(metodo_init_pesos)

        ###########################################################################################
        # definiciones varias
        self.lr = lr
        self.loss_func = nn.MSELoss()
        self.optim = tr.optim.Adam(self.parameters(), lr = self.lr)
        self.device = device
        self.to(device)
        self.entrada_dim = arq[0]
        self.salida_dim = arq[-1]
        self.z_dim = z_dim

        ###########################################################################################
        # tensorboard
        self.base_log_dir = log_dir
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.run_specific_log_dir = os.path.join(self.base_log_dir, current_time)
        self.escritor = SummaryWriter(log_dir = self.run_specific_log_dir)
    
    #########################################################################
    # metodos privados:
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
        for mod in self.encoder:
            if isinstance(mod, nn.Linear):
                metodo_init_pesos(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
        for mod in self.decoder:
            if isinstance(mod, nn.Linear):
                metodo_init_pesos(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
    #########################################################################

    def encode(self, x):
        x = x.view(x.shape[0], -1) # con la forma nueva de construir la red, puede que no vaya mas
        z = self.encoder(x)
        return z

    def decode(self, z):
        xhat = self.decoder(z)
        return xhat

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return xhat

    def fit(self, loader, verbose = False, epoca: int = 0):
        """Función de entrenamiento (una época)"""
        self.train()
        epoch_loss = 0
        if verbose:
            loader = tqdm(loader)
        for x, _ in loader:

            x = x.to(self.device)

            self.optim.zero_grad()
            xhat = self(x)
            loss = self.loss_func(xhat, x)
            epoch_loss += loss.item()
            loss.backward()
            self.optim.step()
        
        perdida_promedio_epoca = epoch_loss / len(loader)
        self.escritor.add_scalar('Perdida/entrenamiento', perdida_promedio_epoca, epoca)

        return epoch_loss/len(loader)


    def test(self, loader, epoca: int):
        """Función de evaluación (una época)"""
        self.eval()
        epoch_loss = 0
        ref, pred = [], []
        for x, _ in loader:
            with tr.no_grad():
                x = x.to(self.device)
                xhat = self(x)

            loss = self.loss_func(xhat, x)
            epoch_loss += loss.item()
            # calculo similaridad coseno y SNR para registrar tambien
            similarity = tr.cosine_similarity(x, xhat, dim=1)
            snr = 10 * tr.log10(tr.sum(x ** 2) / tr.sum((x - xhat) ** 2))

        perdida_promedio_epoca = epoch_loss / len(loader)
        similiaridad_promedio_epoca = similarity.mean().item()
        snr_promedio_epoca = snr.mean().item()

        self.escritor.add_scalar('Perdida/validacion', perdida_promedio_epoca, epoca)
        self.escritor.add_scalar('Similaridad/validacion', similiaridad_promedio_epoca, epoca)
        self.escritor.add_scalar('SNR/validacion', snr_promedio_epoca, epoca)

        return epoch_loss/len(loader)

    def entrenar(self, loader_train, loader_test, n_epochs=10, verbose=False, name_model: str = 'modelo.pmt'):
        """
        Entrena el modelo durante n_epochs.
        """
        log = []
        best_loss, counter, patience = 999, 0, 10
        for epoca in tqdm(range(1, n_epochs + 1),
                          desc = "Entrenamiento total",
                          unit = "época"):
            t0 = time.time()
            loss_train = self.fit(loader_train, verbose, epoca=epoca)
            loss_test = self.test(loader_test, epoca=epoca)

            if loss_test < best_loss:
                best_loss = loss_test
                tr.save(self.state_dict(), name_model)
                counter = 0
            else:
                counter += 1
                if counter > patience:
                    break

            print(f'Epoch {epoca}, train_loss {np.sqrt(loss_train):.6f}, val_loss {np.sqrt(loss_test):.6f} ({time.time()-t0:.2f} s)')
            log.append([loss_train, loss_test])
            self._log_pesos_y_gradientes(epoca)
        self.escritor.close()
        print(f"\nEntrenamiento completado.")
        print(f"Logs guardados en: {self.run_specific_log_dir}")
        print(f"Para ver este run específico en TensorBoard: tensorboard --logdir=\"{self.run_specific_log_dir}\"")
        print(f"Para comparar runs dentro del experimento ({self.base_log_dir}): tensorboard --logdir=\"{self.base_log_dir}\"")

    def _log_pesos_y_gradientes (self, epoca: int):
        '''
        Registra pesos y gradientes en TensorBoard.
        '''
        for nombre, parametro in self.named_parameters():
            self.escritor.add_histogram(f'Parametros/{nombre}', parametro.data.detach().cpu(), epoca)
            if parametro.grad is not None:
                self.escritor.add_histogram(f'Gradientes/{nombre}', parametro.grad.detach().cpu(), epoca)
    
