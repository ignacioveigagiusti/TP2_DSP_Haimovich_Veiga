#%%
#Criterio dee Boll

#Importo librerias
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

fs=48000

#Importo archivos de audio con ruido
signal1, fs = sf.read('ClarinetHiNoise.wav')
signal2, fs = sf.read('GlockHiNoise.wav')
signal3, fs = sf.read('VegaHiNoise.wav')

#Grafico las señales con ruido en funcion del tiempo

signal1 = signal1[:,0] #Magia para que se grafique una sola señal
signal2 = signal2[:,0] #Magia para que se grafique una sola señal
signal3 = signal3[:,0] #Magia para que se grafique una sola señal

t1=np.linspace(0, len(signal1)//fs,len(signal1))
plt.figure(1, figsize=(25,15))
plt.subplot(3,1,1)
plt.grid()
plt.plot(t1, signal1)
plt.title("Clarinete con ruido")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")

plt.subplot(3,1,2)
plt.grid()
plt.plot(t1, signal2)
plt.title("Glock con ruido")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")

plt.subplot(3,1,3)
plt.grid()
plt.plot(t1, signal3)
plt.title("Vega con ruido")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")


# Ventaneo las funciones para identificar los niveles donde solo hay ruido
def mediaMovilD(x, M):
    if len(x)<M:
        raise Exception('La ventana no debe tener más muestras que la señal a filtrar')
    y = np.zeros(len(x))
    for i in range(M//2, len(x) - M//2):
        y[i] = 0.0
        for j in range(-M//2, M//2 + 1):
            y[i] += x[i+j]
        y[i] = y[i] / M
    return y

ventaneo1 = mediaMovilD(signal1,25000)
ventaneo2 = mediaMovilD(signal2,25000)
ventaneo3 = mediaMovilD(signal3,25000)

plt.figure(2, figsize=(25,15))
plt.grid()
plt.plot(t1, ventaneo1)


# Importo funcion short time energy
# def shortTimeEnergy(M,x,hop):
#     if len(x) < (hop-M):
#         raise Exception('El salto entre frames no debe tener más muestras que la señal a filtrar menos la ventana de cada frame')
#     if len(x)<M:
#         raise Exception('La ventana no debe tener más muestras que la señal a filtrar')
#     ste = np.zeros((len(x)-M)//hop)
#     w = np.hamming(M)
#     for i in range(0,((len(x)-M)//hop)):
#         for j in range(0,M):
#             if (j+(i*hop)) < ((len(x)-M+1)):
#                 y = x[j+(i*hop)] * w[j]
#                 ste[i] += ( ((y)**2) / M )   
#     return ste

# energia1 = shortTimeEnergy(2400,signal1,100)

#%%


