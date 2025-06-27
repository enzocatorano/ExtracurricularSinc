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

    entrenamiento, validacion, prueba = datos.particionar(0.7, True, semilla = 123)

    #datos.normalizar()

    batch_size = 16
    cargador_entrenamiento = DataLoader(entrenamiento, batch_size = batch_size, shuffle = True, drop_last = True)
    cargador_validacion = DataLoader(validacion, batch_size = batch_size, shuffle = False)
    cargador_prueba = DataLoader(prueba, batch_size = 1, shuffle = False)

    # Hyperparámetros
    arq_encoder = [entrenamiento[0][0].shape[0]]
    z_dim = 4096
    arq_decoder = [entrenamiento[0][0].shape[0]]
    func_act = 'relu'
    func_act_ultima_capa = False
    usar_batch_norm = True
    dropout = 0.05
    lr = 5e-4
    metodo_init_pesos = tr.nn.init.xavier_normal_
    DEVICE = 'cuda' if tr.cuda.is_available() else 'cpu'
    print(f'Usando dispositivo: {DEVICE}')

    # correr
    net = AE(arq_encoder=arq_encoder,
             z_dim=z_dim,
             arq_decoder=arq_decoder,
             func_act=func_act,
             func_act_ultima_capa=func_act_ultima_capa,
             usar_batch_norm=usar_batch_norm,
             dropout=dropout,
             device=DEVICE,
             log_dir='runs/ae_remastered',
             lr = lr,
             metodo_init_pesos = metodo_init_pesos)
        
    net.entrenar(cargador_entrenamiento, cargador_validacion, n_epochs = 500, name_model = 'ae_sujeto1_4096')
    
    # print(f'Input: {dato_azar_on_device.shape}, Output: {xhat_on_device.shape}')
    print('#####################################################################')
    # en todo el dataset de prueba
    # calculo la similaridad cosen y SNR promedio
    net.eval()
    net.to(DEVICE)
    similarity = []
    snr = []
    for x, _ in cargador_prueba:
        with tr.no_grad():
            x = x.to(DEVICE)
            xhat = net(x)
            similarity.append(tr.cosine_similarity(x, xhat, dim=1))
            snr.append(10 * tr.log10(tr.sum(x ** 2) / tr.sum((x - xhat) ** 2)))
    similarity = tr.cat(similarity)
    snr = tr.stack(snr)
    print('En los datos de prueba')
    print(f'Similaridad coseno promedio: {similarity.mean().item()}')
    print(f'SNR promedio: {snr.mean().item()}')
    print('#####################################################################')


    print('Pruebo dato al azar')
    net.to(DEVICE)
    dato_azar_cpu = prueba[np.random.randint(0, len(prueba))][0] # Shape: [x_size]
    # Add a batch dimension and move to the correct device
    dato_azar_batched_cpu = dato_azar_cpu.unsqueeze(0) # Shape: [1, x_size]
    dato_azar_on_device = dato_azar_batched_cpu.to(DEVICE)

    z = net.encode(dato_azar_on_device)
    xhat_on_device = net.decode(z)

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
    

if __name__ == '__main__':
    main()