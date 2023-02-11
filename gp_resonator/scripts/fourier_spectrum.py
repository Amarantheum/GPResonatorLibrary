import numpy as np
from scipy.signal import find_peaks
from scipy.io.wavfile import read
import sys
from matplotlib import pyplot as plt 

if len(sys.argv) != 2:
    print("expected argument for file location")
    exit()
print("Opening file: " + str(sys.argv[1]))
audio = read(sys.argv[1])
audio = np.array(audio[1],dtype=float)
audio = np.sum(audio, 1) / 2.0
spectrum = np.abs(np.fft.rfft(audio))

f1 = plt.figure(1)
plt.plot(np.arange(0, audio.size), audio)
plt.title("Waveform")
f1 = plt.figure(2)
plt.plot(np.arange(0, spectrum.size), spectrum)
plt.title("Spectrum")
peaks, _ = find_peaks(spectrum,)
print(peaks)
plt.scatter(peaks, spectrum[peaks], color="red")
plt.show()