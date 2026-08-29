import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

try:
    import soundfile as sf
except ImportError:
    sf = None

# Create a sample acoustic signal
sample_rate = 22050
duration = 3
t = np.linspace(0, duration, sample_rate * duration, endpoint=False)

signal = (
    0.6 * np.sin(2 * np.pi * 440 * t) +
    0.3 * np.sin(2 * np.pi * 880 * t) +
    0.1 * np.random.randn(len(t))
)

# Time-domain waveform
plt.figure(figsize=(10, 4))
plt.plot(t, signal)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Audio Time-Domain Waveform")
plt.tight_layout()
plt.savefig("outputs/waveform.png")
plt.close()

# Frequency-domain FFT
freqs = np.fft.rfftfreq(len(signal), 1 / sample_rate)
spectrum = np.abs(np.fft.rfft(signal))

plt.figure(figsize=(10, 4))
plt.plot(freqs, spectrum)
plt.xlim(0, 3000)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency-Domain Spectrum")
plt.tight_layout()
plt.savefig("outputs/frequency_spectrum.png")
plt.close()

# Spectrogram using matplotlib
plt.figure(figsize=(10, 5))
plt.specgram(signal, Fs=sample_rate, NFFT=1024, noverlap=512)
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")
plt.title("Audio Spectrogram")
plt.colorbar(label="Intensity")
plt.tight_layout()
plt.savefig("outputs/spectrogram.png")
plt.close()

# Save generated WAV if soundfile is available
if sf:
    sf.write("outputs/sample_signal.wav", signal, sample_rate)

print("Audio processing completed.")
print("Main frequencies are approximately 440 Hz and 880 Hz.")
print("Check the outputs/ folder for waveform, spectrum, and spectrogram.")
