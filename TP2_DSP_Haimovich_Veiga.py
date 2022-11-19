#%%
#Criterio dee Boll

#Importo librerias
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa

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
def mediamovildr(x,M):
    if len(x)<M:
        raise Exception('La ventana no debe tener más muestras que la señal a filtrar')
    if len(x)>M:
        y = np.zeros(len(x))
        acc=0.0
        for i in range(0,M):
            acc += x[i]
        y[M//2] = acc/M
        for i in range((M//2)+1,(len(y)-(M//2))):
            acc = acc + x[i+((M-1)//2)]-x[i-(((M-1)//2)+1)]
            y[i] = acc/M
        return y
    else:
        s=len(x)-M
        return np.hstack([np.zeros(M-1),np.mean(x[s:s+M-1])])

ventaneo1 = mediamovildr(signal1,25000)
ventaneo2 = mediamovildr(signal2,25000)
ventaneo3 = mediamovildr(signal3,25000)

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

energia1 = librosa.feature.rms(signal1,2400,100)

print(energia1)
plt.figure(3, figsize=(25,15))
plt.grid()
plt.plot(np.linspace(0,len(signal1)//fs,len(energia1[0])), energia1[0])

zeros1 = librosa.feature.zero_crossing_rate(signal1,2400,100)

signal1_noise = []

print(zeros1)
plt.figure(3, figsize=(25,15))
plt.grid()
plt.plot(np.linspace(0,len(signal1)//fs,len(zeros1[0])), zeros1[0])

#%%


