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
plt.figure(1, figsize=(10,7))
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
plt.tight_layout()
plt.show()

# Detección de secciones con ruido.
def onlyNoiseRangeDetection(x):
    # STE de la señal
    energia1 = librosa.feature.rms(y=signal1,frame_length=2400,hop_length=100)
    # Se promedia para tener un punto de referencia para la detección
    avgSTE1 = np.mean(energia1)
    # Detección de actividad vocal/musical
    signal1_VA = np.zeros(len(energia1[0]))
    for i in range(0,len(energia1[0])):
        if energia1[0][i] < (avgSTE1):
            signal1_VA[i]=1
        else:
            signal1_VA[i]=0
    # Extracción de rangos sin actividad y búsqueda del rango más largo
    noiseRanges = [[0,0]]
    nOfRanges = 0
    longestRange = 0
    longestRange_len = 0
    for i in range(0,len(signal1_VA)):
        if signal1_VA[i]==1:
            noiseRanges[nOfRanges][1] = i 
        else:
            if (noiseRanges[nOfRanges][1] - noiseRanges[nOfRanges][0]) > longestRange_len:
                longestRange = nOfRanges
                longestRange_len = noiseRanges[nOfRanges][1] - noiseRanges[nOfRanges][0]
            nOfRanges += 1
            noiseRanges.append([i,i])
    return x[noiseRanges[longestRange][0]:noiseRanges[longestRange][1]]

def spectralSubtraction(x,M,hop):
    # STFT de x
    Y = librosa.stft(x)
    # Extracción de Magnitud y Fase
    Y_abs = np.abs(Y)
    Y_phase = np.angle(Y)
    # Selección de rango de la señal que contenga sólo ruido
    noise=onlyNoiseRangeDetection(x)
    # Promedio espectral del ruido (mu)
    mu=np.mean(abs(fft(noise)))
    # Estimador de la magnitud de la señal original
    S_squared = Y_abs**2 - mu**2
    # Rectificación de media onda
    for i in range(len(S_squared)):
        for j in range(len(S_squared[i])):
            if S_squared[i][j]<0:
                S_squared[i][j]=0
    S_half_wave = np.sqrt(S_squared)*np.exp(1j*Y_phase)
    x_half_wave = librosa.istft(S_half_wave)
    # Ruido residual
    noise_res =onlyNoiseRangeDetection(x_half_wave)
    N_r = np.amax(abs(fft(noise_res)))
    # Eliminación mediante selección de valores mínimos adyacentes
    for i in range(len(S_squared)):
        for j in range(1,len(S_squared[i])-1):
            if np.sqrt(S_squared[i][j])<N_r:
                S_squared[i][j]=np.amin([S_squared[i][j-1],S_squared[i][j],S_squared[i][j+1]])
    S_res = np.sqrt(S_squared)*np.exp(1j*Y_phase)
    x_res = librosa.istft(S_half_wave)
    # Ruido residual 2
    noise_res_2 =onlyNoiseRangeDetection(x_res)
    mu_res_2 = np.mean(abs(fft(noise_res_2)))
    # Atenuación aditional en secciones de sólo ruido
    for i in range(len(S_squared)):
        if np.sum(S_squared[i])>0:
            T=20*np.log10((1/(2*np.pi))*np.sum(np.sqrt(S_squared[i])/mu_res_2))
            if T<-12:
                c=10**(-30/20)
                S_squared[i]=c*S_squared[i]
    # Incorporación de fase y transformada inversa
    S_f = np.sqrt(S_squared)*np.exp(1j*Y_phase)
    noiseless_s = librosa.istft(S_f)
    return noiseless_s

signal1_boll = spectralSubtraction(signal1,1536,768)

signal2_boll = spectralSubtraction(signal2,1536,768)

signal3_boll = spectralSubtraction(signal3,1536,768)

plt.figure(2, figsize=(10,7))
plt.subplot(3,1,1)
plt.grid()
plt.plot(t1[:len(signal1_boll)], signal1_boll)
plt.title("Clarinete sin ruido (Boll)")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")

plt.subplot(3,1,2)
plt.grid()
plt.plot(t1[:len(signal2_boll)], signal2_boll)
plt.title("Glock sin ruido (Boll)")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")

plt.subplot(3,1,3)
plt.grid()
plt.plot(t1[:len(signal3_boll)], signal3_boll)
plt.title("Vega sin ruido (Boll)")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.tight_layout()
plt.show()

sf.write('Clarinet_boll.wav',signal1_boll,fs)
sf.write('Glock_boll.wav',signal2_boll,fs)
sf.write('Vega_boll.wav',signal3_boll,fs)

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
    x_p = np.mean(abs(x))**2
    mu_p = mu**2
    if mu_p == 0:
        return ((x_p)/0.001)
    SNR = ((x_p)/mu_p)
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
    noise=onlyNoiseRangeDetection(x)
    # Defino el parámetro de piso espectral Beta
    beta = 0.03
    # Estimador de la magnitud de la señal original calculado por bandas
    Y_t = Y.transpose()
    S_squared_t = np.zeros_like(Y_t)
    for i in range(len(Y_t)):
        for k in range(bands):
            band_start = k*(len(Y_t[i])//(bands+1))
            band_end = (k+2)*(len(Y_t[i])//(bands+1))
            band_i = Y_t[i][band_start:band_end]
            band_i = band_i*np.hanning(len(band_i))
            # Promedio espectral del ruido de la banda (mu)
            mu_i=np.mean(abs(fft(noise[band_start:band_end])))
            for j in range(len(band_i)):
                Y_p = band_i[j]**2
                alpha = alpha_i(band_i, mu_i)
                delta = delta_i(band_end, fs)
                S_squared_t[i][j+(k*len(band_i)//2)] += Y_p - (alpha*delta*(mu_i**2))
                if (S_squared_t[i][j+(k*len(band_i)//2)]) < (beta*Y_p):
                    S_squared_t[i][j+(k*len(band_i)//2)] += beta*(Y_p)
    S_squared = S_squared_t.transpose()
    S_half_wave = np.sqrt(S_squared)*np.exp(1j*Y_phase)
    x_half_wave = librosa.istft(S_half_wave)
    # Ruido residual
    noise_res =onlyNoiseRangeDetection(x_half_wave)
    N_r = np.amax(abs(fft(noise_res)))
    # Eliminación mediante selección de valores mínimos adyacentes
    for i in range(len(S_squared)):
        for j in range(1,len(S_squared[i])-1):
            if np.sqrt(S_squared[i][j])<N_r:
                S_squared[i][j]=np.amin([S_squared[i][j-1],S_squared[i][j],S_squared[i][j+1]])
    S_res = np.sqrt(S_squared)*np.exp(1j*Y_phase)
    x_res = librosa.istft(S_half_wave)
    # Ruido residual 2
    noise_res_2 =onlyNoiseRangeDetection(x_res)
    mu_res_2 = np.mean(abs(fft(noise_res_2)))
    # Atenuación aditional en secciones de sólo ruido
    for i in range(len(S_squared)):
        if np.sum(S_squared[i])>0:
            T=20*np.log10((1/(2*np.pi))*np.sum(np.sqrt(S_squared[i])/mu_res_2))
            if T<-12:
                c=10**(-30/20)
                S_squared[i]=c*S_squared[i]
    # Incorporación de fase y transformada inversa
    S_f = np.sqrt(S_squared)*np.exp(1j*Y_phase)
    noiseless_s = librosa.istft(Y_t.transpose())
    return noiseless_s

signal1_MBSS = MBSS(signal1,1536,768,4)

signal2_MBSS = MBSS(signal2,1536,768,4)

signal3_MBSS = MBSS(signal3,1536,768,4)

plt.figure(2, figsize=(10,7))
plt.subplot(3,1,1)
plt.grid()
plt.plot(t1[:len(signal1_MBSS)], signal1_MBSS)
plt.title("Clarinete sin ruido (MBSS)")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")

plt.subplot(3,1,2)
plt.grid()
plt.plot(t1[:len(signal2_MBSS)], signal2_MBSS)
plt.title("Glock sin ruido (MBSS)")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")

plt.subplot(3,1,3)
plt.grid()
plt.plot(t1[:len(signal3_MBSS)], signal3_MBSS)
plt.title("Vega sin ruido (MBSS)")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.tight_layout()
plt.show()

sf.write('Clarinet_MBSS.wav',signal1_MBSS,fs)
sf.write('Glock_MBSS.wav',signal2_MBSS,fs)
sf.write('Vega_MBSS.wav',signal3_MBSS,fs)

#%%

# Estimación de SNRs

def SNR_estimate(noisySignal, noiseSubtractedSignal):
    mu = np.mean(abs(noiseSubtractedSignal))**2
    noise = onlyNoiseRangeDetection(noisySignal)
    sigma = np.std(abs(noise))**2
    SNR = mu/sigma
    return SNR

SNR_Boll_Clarinet = SNR_estimate(signal1, signal1_boll)
SNR_MBSS_Clarinet = SNR_estimate(signal1, signal1_MBSS)
SNR_Boll_Glock = SNR_estimate(signal2, signal2_boll)
SNR_MBSS_Glock = SNR_estimate(signal2, signal2_MBSS)
SNR_Boll_Vega = SNR_estimate(signal3, signal3_boll)
SNR_MBSS_Vega = SNR_estimate(signal3, signal3_MBSS)

print(SNR_Boll_Clarinet)
print(SNR_MBSS_Clarinet)
print(SNR_Boll_Glock)
print(SNR_MBSS_Glock)
print(SNR_Boll_Vega)
print(SNR_MBSS_Vega)

# SNR_Boll_Clarinet = 57.89421936716158
# SNR_MBSS_Clarinet = 58.363355683136156
# SNR_Boll_Glock = 10.650011505113806
# SNR_MBSS_Glock = 10.979366382341569
# SNR_Boll_Vega = 15.104270107567498
# SNR_MBSS_Vega = 15.537580503859074
