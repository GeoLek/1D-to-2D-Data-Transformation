import mne
import matplotlib.pyplot as plt

# Load the EDF file (replace 'path_to_your_file.edf' with your file path)
raw = mne.io.read_raw_edf('path_to_your_file.edf', preload=True)

# Plot time series for all channels
raw.plot(scalings='auto', title='EEG Time Series', show=True, block=True)

# Plot the power spectral density (PSD) for each channel
raw.plot_psd(fmin=0.5, fmax=50, average=True)
plt.title("Power Spectral Density (PSD)")

# Plot topographic sensor layout (if channel locations are available in the EDF file)
raw.plot_sensors(show_names=True)
plt.title("EEG Sensor Layout")

plt.show()
