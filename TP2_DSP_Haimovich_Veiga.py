#%%

# Resta Espectral Boll

#Importo librerias
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import librosa

fs=48000

#Importo archivos de audio con ruido
signal1, fs = sf.read('ClarinetHiNoise.wav')
signal2, fs = sf.read('GlockHiNoise.wav')
signal3, fs = sf.read('VegaHiNoise.wav')

#Grafico las señales con ruido en funcion del tiempo

signal1 = signal1[:,0] 
signal2 = signal2[:,0] 
signal3 = signal3[:,0] 

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

def spectralSubtraction(x,M,hop):
    # STFT de x
    Y = librosa.stft(x)
    # Extracción de Magnitud y Fase
    Y_abs = np.abs(Y)
    Y_phase = np.angle(Y)
    # Selección de rango de la señal que contenga sólo ruido
    noise=np.array(x[0:int(5*fs)])
    # Promedio espectral del ruido (mu)
    mu=np.mean(abs(fft(noise)))
    # Estimador de la magnitud de la señal original
    S_squared = Y_abs**2 - mu**2
    # Rectificación de media onda
    for i in range(len(S_squared)):
        for j in range(len(S_squared[i])):
            if S_squared[i][j]<0:
                S_squared[i][j]=0
    # Incorporación de fase y transformada inversa
    S_f = np.sqrt(S_squared)*np.exp(1j*Y_phase)
    noiseless_s = librosa.istft(S_f)
    return noiseless_s

signal1_boll = spectralSubtraction(signal1,1536,768)

signal2_boll = spectralSubtraction(signal2,1536,768)

signal3_boll = spectralSubtraction(signal3,1536,768)

plt.figure(2, figsize=(25,15))
plt.subplot(3,1,1)
plt.grid()
plt.plot(t1[:len(signal1_boll)], signal1_boll)
plt.title("Clarinete sin ruido")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")

plt.subplot(3,1,2)
plt.grid()
plt.plot(t1[:len(signal2_boll)], signal2_boll)
plt.title("Glock sin ruido")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")

plt.subplot(3,1,3)
plt.grid()
plt.plot(t1[:len(signal3_boll)], signal3_boll)
plt.title("Vega sin ruido")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")

sf.write('signal1_boll.wav',signal1_boll,fs)
sf.write('signal2_boll.wav',signal2_boll,fs)
sf.write('signal3_boll.wav',signal3_boll,fs)

#%%

# Resta Espectral Multibanda

#Importo librerias
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import librosa

fs=48000

#Importo archivos de audio con ruido
signal1, fs = sf.read('ClarinetHiNoise.wav')
signal2, fs = sf.read('GlockHiNoise.wav')
signal3, fs = sf.read('VegaHiNoise.wav')

# Se toma 1 sólo canal
signal1 = signal1[:,0] 
signal2 = signal2[:,0] 
signal3 = signal3[:,0] 

t1=np.linspace(0, len(signal1)//fs,len(signal1))

def SNR(x,mu):
    x_p = np.mean(x)**2
    mu_p = mu**2
    SNR = (x_p)/mu_p
    return SNR

def alpha_i(x,mu):
    if SNR(x,mu)<-5:
        return 5
    if SNR(x,mu)>-5:
        if SNR(x,mu)<20:
            return (3/20)*SNR(x,mu)
        else:
            return 1

def delta_i(f,fs):
    if f<1000:
        return 0.8
    if f>1000:
        if f<((fs//2)-2000):
            return 1.3
        else:
            return 1

def MBSS(x,M,hop,bands):
    # STFT de x
    Y = librosa.stft(x)
    # Extracción de Magnitud y Fase
    Y_abs = np.abs(Y)
    Y_phase = np.angle(Y)
    # Selección de rango de la señal que contenga sólo ruido
    noise=np.array(x[0:int(5*fs)])
    # Promedio espectral del ruido (mu)
    mu=np.mean(abs(fft(noise)))
    # Defino el parámetro de piso espectral Beta
    beta = 0.02
    # Estimador de la magnitud de la señal original calculado por bandas
    S_squared = np.zeros_like(Y)
    for i in range(bands):
        for j in range(len(Y[0])):
            for k in range(i*(len(Y)//bands),(i+1)*(len(Y)//bands)):
                Y_p = Y[k][j]**2
                alpha = alpha_i(Y[i*(len(Y)//bands):(i+1)*(len(Y)//bands),j], mu)
                delta = delta_i((fs/len(Y))*(i+1)*(len(Y)//bands), fs)
                S_squared[k][j] = Y_p - (alpha*delta*(mu**2))
                if (S_squared[k][j]) < (beta*Y_p):
                    S_squared[k][j]=beta*(Y_p)
    # Incorporación de fase y transformada inversa
    S_f = np.sqrt(S_squared)*np.exp(1j*Y_phase)
    noiseless_s = librosa.istft(S_f)
    return noiseless_s

signal1_MBSS = MBSS(signal1,1536,768,4)

signal2_MBSS = MBSS(signal2,1536,768,4)

signal3_MBSS = MBSS(signal3,1536,768,4)

plt.figure(2, figsize=(25,15))
plt.subplot(3,1,1)
plt.grid()
plt.plot(t1[:len(signal1_MBSS)], signal1_MBSS)
plt.title("Clarinete sin ruido")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")

plt.subplot(3,1,2)
plt.grid()
plt.plot(t1[:len(signal2_MBSS)], signal2_MBSS)
plt.title("Glock sin ruido")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")

plt.subplot(3,1,3)
plt.grid()
plt.plot(t1[:len(signal3_MBSS)], signal3_MBSS)
plt.title("Vega sin ruido")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")

sf.write('signal1_MBSS.wav',signal1_MBSS,fs)
sf.write('signal2_MBSS.wav',signal2_MBSS,fs)
sf.write('signal3_MBSS.wav',signal3_MBSS,fs)


#%%

# Detección de secciones con ruido.

energia1 = librosa.feature.rms(signal1,2400,100)
avgSTE1 = np.mean(energia1)
signal1_VA = np.zeros(len(energia1[0]))
for i in range(0,len(energia1[0])):
    if energia1[0][i] < (avgSTE1):
        signal1_VA[i]=1
    else:
        signal1_VA[i]=0

noiseRanges = [[0,0]]
nOfRanges = 0
longestRange = 0
for i in range(0,len(signal1_VA)):
    if signal1_VA[i]==1:
       noiseRanges[nOfRanges][1] = i 
    else:
        if (noiseRanges[nOfRanges][1] - noiseRanges[nOfRanges][0]) > longestRange:
            longestRange = nOfRanges
        nOfRanges += 1
        noiseRanges.append([i,i])

print(noiseRanges)

print(longestRange)
print(noiseRanges[longestRange])

plt.figure(3, figsize=(25,15))
plt.grid()
plt.plot(np.linspace(0,len(signal1)//fs,len(energia1[0])), energia1[0])
plt.plot(np.linspace(0,len(signal1)//fs,len(energia1[0])), signal1_VA)

zeros1 = librosa.feature.zero_crossing_rate(signal1,500,100)

signal1_noise = []

print(zeros1)
plt.figure(4, figsize=(25,15))
plt.grid()
plt.plot(np.linspace(0,len(signal1)//fs,len(zeros1[0])), zeros1[0])

# %%
