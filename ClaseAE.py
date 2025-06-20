from torch import nn
import torch as tr
from tqdm import tqdm
import time
import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np

class AE(nn.Module):
    def __init__(self, emb_size, imsize=[1, 26], device='cpu', log_dir: str = 'runs/ae', lr=0.001):
        super().__init__()

        nin = imsize[0]*imsize[1]
        self.encoder = nn.Sequential(nn.Linear(nin, emb_size),
                                     nn.Sigmoid())

        self.decoder = nn.Sequential(nn.Linear(emb_size, nin))
        self.lr = lr
        self.loss_func = nn.MSELoss()
        self.optim = tr.optim.Adam(self.parameters(), lr= self.lr)
        self.device = device
        self.to(device)
        self.imsize = imsize
        self.base_log_dir = log_dir
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.run_specific_log_dir = os.path.join(self.base_log_dir, current_time)
        self.escritor = SummaryWriter(log_dir=self.run_specific_log_dir)

    def encode(self, x):
        x = x.view(x.shape[0], -1)
        z = self.encoder(x)
        return z

    def decode(self, z):
        xhat = self.decoder(z)
        return xhat

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return xhat

    def fit(self, loader, verbose=False, epoca: int = 0):
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
        epoch_loss = 0
        ref, pred = [], []
        for x, _ in loader:
            with tr.no_grad():
                x = x.to(self.device)
                xhat = self(x)

            loss = self.loss_func(xhat, x)
            epoch_loss += loss.item()
        perdida_promedio_epoca = epoch_loss / len(loader)

        self.escritor.add_scalar('Perdida/validacion', perdida_promedio_epoca, epoca)
        return epoch_loss/len(loader)

    def entrenar(self, loader_train, loader_test, n_epochs=10, verbose=False, name_model: str = 'modelo.pmt'):
        """
        Entrena el modelo durante n_epochs.
        """
        log = []
        best_loss, counter, patience = 999, 0, 3
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
    
