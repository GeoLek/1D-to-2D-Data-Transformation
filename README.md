# 1D to 2D Data Transformation

The repository for the published paper 'Time-Frequency Transformations for Enhanced Biomedical Signal Classification with CNNs'

In this initiative we investigated the effectiveness of six different 1D-to-2D transformation methods: Continuous Wavelet Transform (CWT), Discrete Fourier Transform (DFT), Fast Fourier Transform (FFT), Short-Time Fourier Transform (STFT), Signal Reshaping (SR), and Recurrence Plots (RPs) to classify electrocardiograms (ECGs) and electroencephalograms (EEGs) signals.

# Datasets
For ECG signals we used the MIT-BIH Arrhythmia Dataset from PhysioNet (Version: 1.0.0). Find it [here](https://physionet.org/content/mitdb/1.0.0/)  
For EEG signals we used the Epilepsy EEG Dataset (University of Bonn). Find it [here](https://www.ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads/)

# Preprocessing steps for ECG & EEG signals

![Screenshot 2024-12-04 00:12:31](https://github.com/user-attachments/assets/251173f3-4def-440b-b4ee-a6f25d9b7b37)


# Results of 1D-to-2D Transformations for ECG and EEG Signals

### Classification Results for ECG Signals
| Dimension Transformation Method | Accuracy | F1-Score | Recall | Precision |
|---------------------------------|----------|----------|--------|-----------|
| **Minimal 2D CNN**              |          |          |        |           |
| Continuous Wavelet Transform (CWT) | 0.98     | 0.98     | 0.98   | 0.98      |
| Discrete Fourier Transform (DFT)   | 0.87     | 0.87     | 0.87   | 0.87      |
| Fast Fourier Transform (FFT)       | 0.88     | 0.88     | 0.88   | 0.88      |
| Short-Time Fourier Transform (STFT)| 0.95     | 0.95     | 0.95   | 0.95      |
| Recurrence Plots (RPs)             | 0.99     | 0.99     | 0.99   | 0.99      |
| Signal Reshaping (SR)              | 0.96     | 0.96     | 0.96   | 0.96      |
| **LeNet-5 2D CNN**                 |          |          |        |           |
| Continuous Wavelet Transform (CWT) | 0.99     | 0.99     | 0.99   | 0.99      |
| Discrete Fourier Transform (DFT)   | 0.97     | 0.97     | 0.97   | 0.97      |
| Fast Fourier Transform (FFT)       | 0.97     | 0.97     | 0.97   | 0.97      |
| Short-Time Fourier Transform (STFT)| 0.99     | 0.99     | 0.99   | 0.99      |
| Recurrence Plots (RPs)             | 0.99     | 0.99     | 0.99   | 0.99      |
| Signal Reshaping (SR)              | 0.96     | 0.96     | 0.96   | 0.96      |

### Classification Results for EEG Signals
| Dimension Transformation Method | Accuracy | F1-Score | Recall | Precision |
|---------------------------------|----------|----------|--------|-----------|
| **Minimal 2D CNN**              |          |          |        |           |
| Continuous Wavelet Transform (CWT) | 1        | 1        | 1      | 1         |
| Discrete Fourier Transform (DFT)   | 0.83     | 0.83     | 0.83   | 0.83      |
| Fast Fourier Transform (FFT)       | 0.87     | 0.87     | 0.87   | 0.87      |
| Short-Time Fourier Transform (STFT)| 1        | 1        | 1      | 1         |
| Recurrence Plots (RPs)             | 1        | 1        | 1      | 1         |
| Signal Reshaping (SR)              | 0.77     | 0.77     | 0.77   | 0.77      |
| **LeNet-5 2D CNN**                 |          |          |        |           |
| Continuous Wavelet Transform (CWT) | 0.99     | 0.99     | 0.99   | 0.99      |
| Discrete Fourier Transform (DFT)   | 0.97     | 0.97     | 0.97   | 0.97      |
| Fast Fourier Transform (FFT)       | 0.97     | 0.97     | 0.97   | 0.97      |
| Short-Time Fourier Transform (STFT)| 0.99     | 0.99     | 0.99   | 0.99      |
| Recurrence Plots (RPs)             | 1        | 1        | 1      | 1         |
| Signal Reshaping (SR)              | 0.96     | 0.96     | 0.96   | 0.96      |
