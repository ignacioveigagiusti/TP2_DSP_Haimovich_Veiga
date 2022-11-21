#%%
#Criterio de Boll

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

#Para realizar la resta espectral tengo que obtener la transformada de la magnitud de la señal original ventaneada y el promedio espectral del ruido

def windowing(M,x,hop):
    if len(x) < (hop-M):
        raise Exception('El salto entre frames no debe tener más muestras que la señal a filtrar menos la ventana de cada frame')
    if len(x)<M:
        raise Exception('La ventana no debe tener más muestras que la señal a filtrar')
    w = np.hanning(M)

    # Creo vector que contendrá a cada frame
    windowedFrames = np.zeros(((len(x)//hop)-1,M))

    for i in range(0,(int((len(x)-M)//hop))):
        windowedFrames[i] = x[(i*hop):(i*hop+M)] * w
    
    return windowedFrames

M= 1536
hop = 768

def noiseReductionBoll(x,M,hop):
    # Ventaneo de señal
    x_window = windowing(M,x,hop)
    # FFT de cada ventana
    x_window_f = np.zeros((len(x_window),M), dtype='complex_')
    x_window_phase = np.zeros((len(x_window),M), dtype='complex_')
    for i in range(0,len(x_window)):
        x_window_f[i] = fft(x_window[i])
        x_window_phase[i] = np.angle(x_window_f[i])
    #Defino un vector de ruido, eligiendo los primeros 5 seg de la muestra donde hay solo ruido
    noise=np.array(x[0:int(5*fs)])
    #Calculo el promedio espectral del ruido "mu"
    mu=np.mean(abs(fft(noise)))
    # Rectificación de media onda
    S_i = np.zeros((len(x_window),M), dtype='complex_')
    for i in range(0,len(x_window)):
        X_i = fft(x_window[i])
        for j in range(0,len(X_i)):
            if X_i[j] == 0:
                X_i[j]=0.01
        H = 1 - (mu/abs(X_i))
        H_r = (H+abs(H))/2
        S_i[i] = H_r*X_i
    
    # Atenuación cuando no hay voz.
    # T_per_frame = np.zeros(len(x_window_f))
    # for i in range(0,len(x_window_f)):
    #     T_per_frame[i] = 20 * np.log10((1/(M*mu))*np.sum(abs(S_i[i])))
    #     c = 10**(-30/20)
    #     if T_per_frame[i] < -12:
    #         S_i[i] = c*fft(x_window[i])

    x_sin_ruido = np.zeros(len(x))

    for i in range(0,len(x_window_f)):
        S_i_m = ifft(S_i[i]* np.exp(1j*x_window_phase[i]))
        for j in range(0,M):
            x_sin_ruido[i*hop+j] += S_i_m[j]

    return x_sin_ruido

#Antitransformo la señal con parte del ruido sustraído

signal1_boll = noiseReductionBoll(signal1,1536,768)

signal2_boll = noiseReductionBoll(signal2,1536,768)

signal3_boll = noiseReductionBoll(signal3,1536,768)

plt.figure(2, figsize=(25,15))
plt.subplot(3,1,1)
plt.grid()
plt.plot(t1, signal1_boll)
plt.title("Clarinete sin ruido")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")

plt.subplot(3,1,2)
plt.grid()
plt.plot(t1, signal2_boll)
plt.title("Glock sin ruido")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")

plt.subplot(3,1,3)
plt.grid()
plt.plot(t1, signal3_boll)
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

M= 1536
hop = 768

def SNR(x,mu):
    x_p = np.mean(x)**2
    mu_p = mu**2
    SNR = (x_p-mu_p)/mu_p
    return SNR

def alpha_i(x,mu):
    if SNR(x,mu)<-5:
        return 4.75
    if SNR(x,mu)>-5:
        if SNR(x,mu)<20:
            return (3/20)*SNR(x,mu)
        else:
            return 1

def delta_i(f,fs):
    if f<1000:
        return 1
    if f>1000:
        if f<((fs//2)-2000):
            return 2.5
        else:
            return 1.5

def noiseReductionMBSS(x,M,hop,bands):
    # Ventaneo de señal
    x_window = windowing(M,x,hop)
    # FFT de cada ventana
    x_window_f = np.zeros((len(x_window),M), dtype='complex_')
    x_window_phase = np.zeros((len(x_window),M), dtype='complex_')
    for i in range(0,len(x_window)):
        x_window_f[i] = fft(x_window[i])
        x_window_phase[i] = np.angle(x_window_f[i])
    #Defino un vector de ruido, eligiendo los primeros 5 seg de la muestra donde hay solo ruido
    noise=np.array(x[0:int(5*fs)])
    #Calculo el promedio espectral del ruido "mu"
    mu=np.mean(abs(fft(noise)))
    # ventaneo cada FFT separando en 64 bandas con superposición del 50%
    x_window_f_bands = np.zeros((len(x_window),(M//(M//(bands)))-1,M//(bands//2)))
    for i in range(len(x_window_f)):
        x_window_f_bands[i] = windowing(M//32, x_window_f[i], M//64)
    # 
    S_i = np.zeros((len(x_window_f_bands),(M//(M//64))-1,M//32), dtype='complex_')
    for i in range(0,len(x_window_f_bands)):
        for j in range (0,(M//(M//64)-1)):
            for k in range(M//32):
                S_i[i][j][k] = x_window_f_bands[i][j][k]**2 - (alpha_i(x_window_f_bands[i][j], mu)*delta_i((fs/len(x_window_f[i]))*(j+1)*(M//32), fs)*mu**2)
    
    # Atenuación cuando no hay voz.
    # T_per_frame = np.zeros(len(x_window_f))
    # for i in range(0,len(x_window_f)):
    #     T_per_frame[i] = 20 * np.log10((1/(M*mu))*np.sum(abs(S_i[i])))
    #     c = 10**(-30/20)
    #     if T_per_frame[i] < -12:
    #         S_i[i] = c*fft(x_window[i])

    x_sin_ruido = np.zeros(len(x))

    for i in range(0,len(x_window_f_bands)):
        S_k = np.zeros(M)
        for k in range(0,(M//(M//64))-1):
            for l in range(0,M//32):
                if((k*(M//64)+l)<M):
                    S_k[k*(M//64)+l] += S_i[i][k][l]
        S_i_m = ifft(S_k* np.exp(1j*x_window_phase[i]))
        for j in range(0,M):
            x_sin_ruido[i*hop+j] += S_i_m[j]

    return x_sin_ruido

#Antitransformo la señal con parte del ruido sustraído

signal1_MBSS = noiseReductionMBSS(signal1,1536,768,64)

signal2_MBSS = noiseReductionMBSS(signal2,1536,768,64)

signal3_MBSS = noiseReductionMBSS(signal3,1536,768,64)

plt.figure(2, figsize=(25,15))
plt.subplot(3,1,1)
plt.grid()
plt.plot(t1, signal1_boll)
plt.title("Clarinete sin ruido")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")

plt.subplot(3,1,2)
plt.grid()
plt.plot(t1, signal2_boll)
plt.title("Glock sin ruido")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")

plt.subplot(3,1,3)
plt.grid()
plt.plot(t1, signal3_boll)
plt.title("Vega sin ruido")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")

sf.write('signal1_MBSS.wav',signal1_MBSS,fs)
sf.write('signal2_MBSS.wav',signal2_MBSS,fs)
sf.write('signal3_MBSS.wav',signal3_MBSS,fs)


#%%

# Detección de secciones con ruido.

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
        if noiseRanges[nOfRanges][1] - noiseRanges[nOfRanges][0] > longestRange:
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
