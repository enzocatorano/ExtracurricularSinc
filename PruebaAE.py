from ClaseAE import AE
import torch as tr
import time
import buenas_practicas as mbp
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


def prueba_ae():
    sujeto = 1
    prediccion = 'estimulo'
    datos = mbp.DataSetEEG(sujeto)
    datos.dejar_etiqueta(prediccion)
    datos_vocales, datos_comandos = datos.split_estimulo_datasets()

    entrenamiento, validacion, prueba = datos.particionar(0.7, True)

    print(f'Entrenamiento: {len(entrenamiento)} muestras')
    print(entrenamiento[0][0].shape)
    # Entrenamiento: 481 muestras
    # torch.Size([24576])
    # (tensor[...], etiqueta)
    # entrenamiento[0][0] = tensor(...)
    # entrenamiento[0][1] = etiqueta
    


def main():

    

    sujeto = 1
    prediccion = 'estimulo'
    datos = mbp.DataSetEEG(sujeto)
    datos.dejar_etiqueta(prediccion)
    datos_vocales, datos_comandos = datos.split_estimulo_datasets()

    entrenamiento, validacion, prueba = datos.particionar(0.7, True)

    


    #datos.normalizar()

    batch_size = 16
    cargador_entrenamiento = DataLoader(entrenamiento, batch_size = batch_size, shuffle = True, drop_last = False)
    cargador_validacion = DataLoader(validacion, batch_size = batch_size, shuffle = False)
    cargador_prueba = DataLoader(prueba, batch_size = 1, shuffle = False)

    

    # Hyperparámetros
    x_size = entrenamiento[0][0].shape[0]
    train_loader = cargador_entrenamiento
    test_loader = cargador_validacion


    EMB_SIZE = 1024
    DEVICE = 'cuda' if tr.cuda.is_available() else 'cpu'
    print(f'Usando dispositivo: {DEVICE}')
    net = AE(emb_size=EMB_SIZE, imsize = [1, x_size], device=DEVICE, lr = 1e-3)
    
    net.entrenar(train_loader, test_loader, n_epochs = 500, name_model = 'ae_sujeto1')

    net.to(DEVICE)
    dato_azar_cpu = prueba[np.random.randint(0, len(prueba))][0] # Shape: [x_size]
    # Add a batch dimension and move to the correct device
    dato_azar_batched_cpu = dato_azar_cpu.unsqueeze(0) # Shape: [1, x_size]
    dato_azar_on_device = dato_azar_batched_cpu.to(DEVICE)

    z = net.encode(dato_azar_on_device)
    xhat_on_device = net.decode(z)
    
    print(f'Input: {dato_azar_on_device.shape}, Output: {xhat_on_device.shape}')
    print('#####################################################################')
    # Compare along feature dimension (dim=1 for [batch_size, features])
    # Ensure both tensors are on the same device (they are: DEVICE)
    similarity = tr.cosine_similarity(dato_azar_on_device, xhat_on_device, dim=1)
    print(f'Similaridad: {similarity.item()}') # Use .item() for a single scalar value
    
    plt.style.use('dark_background')
    plt.figure(figsize = (18, 6))
    plt.plot(dato_azar_on_device.squeeze().cpu().detach().numpy(), color = 'dodgerblue', label = 'Original')
    plt.plot(xhat_on_device.squeeze().cpu().detach().numpy(), color = 'orange', label = 'Reconstruida')
    #plt.ylim([0, 1])
    plt.grid(linewidth = 0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

    

if __name__ == '__main__':
    main()