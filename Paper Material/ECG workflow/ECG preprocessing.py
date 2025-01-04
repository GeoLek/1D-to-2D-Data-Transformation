import wfdb
import numpy as np
import pandas as pd
import os
from scipy.signal import medfilt, detrend

# Define the directory containing the MIT-BIH data files
data_dir = '/home/orion/Geo/Projects/1D-to-2D-Data-Transformation/MIT-BIH Arrhythmia Database/mit-bih-arrhythmia-database-1.0.0'
output_dir = '/home/orion/Geo/Projects/1D-to-2D-Data-Transformation/Output_updated/ECG/Extracted_Processed_Beats'  # Specify your output directory

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List of record numbers (e.g., 100, 101, 102, ...)
record_numbers = list(range(100, 235))

##############################################################################
# Annotation Mapping
##############################################################################
# Extended mapping of individual symbols to the 5 final classes:
# N: Normal (N, L, R, e, j)
# S: Supraventricular ectopic (A, a, J, S)
# V: Ventricular ectopic (V, E)
# F: Fusion (F)
# Q: Unknown or Unclassifiable (/ , f, Q)
annotation_equiv = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q'
}

# Mapping from our consolidated symbol to a human-readable description:
annotation_desc = {
    'N': 'Normal beat',
    'S': 'Supraventricular ectopic beat',
    'V': 'Ventricular ectopic beat',
    'F': 'Fusion beat',
    'Q': 'Unknown beat'
}

##############################################################################
# Preprocessing functions
##############################################################################
def filter_noise(ecg_signal, kernel_size=3):
    return medfilt(ecg_signal, kernel_size=kernel_size)

def remove_baseline_wander(ecg_signal):
    return detrend(ecg_signal, type='linear')

def normalize_signal(ecg_signal):
    mean_val = np.mean(ecg_signal)
    std_val = np.std(ecg_signal)
    return (ecg_signal - mean_val) / std_val

def preprocess_ecg_signal(ecg_data):
    # Preprocess each lead in the DataFrame
    for column in ecg_data.columns[5:]:  # Skipping first five columns (Sample index, Time, etc.)
        ecg_data[column] = filter_noise(ecg_data[column])
        ecg_data[column] = remove_baseline_wander(ecg_data[column])
        ecg_data[column] = normalize_signal(ecg_data[column])
    return ecg_data

##############################################################################
# Beat extraction
##############################################################################
def extract_beats(record, annotations, signal, time_ms, window_size=100):
    beat_segments = []
    ann_sample_indices = annotations.sample
    ann_symbols = annotations.symbol

    for idx, symbol in zip(ann_sample_indices, ann_symbols):
        # Skip if the symbol is not in the known mapping
        if symbol not in annotation_equiv:
            continue

        # Ensure we can extract a valid window around the beat
        if idx - window_size >= 0 and idx + window_size < len(signal):
            beat_segment = signal[idx - window_size: idx + window_size + 1]
            beat_time_ms = time_ms[idx - window_size: idx + window_size + 1]
            beat_segments.append((beat_segment, beat_time_ms, symbol))

    return beat_segments

##############################################################################
# Main function to process each record
##############################################################################
def process_record(record_number):
    record_name = f'{record_number:03d}'  # Format record number with leading zeros

    try:
        # Load the header file to get channel names
        header = wfdb.rdheader(os.path.join(data_dir, record_name))
        print(f"Columns from header file for record {record_name}: {header.sig_name}")

        # Load the ECG signal data
        record = wfdb.rdrecord(os.path.join(data_dir, record_name))
        signal = record.p_signal[:, 0]  # Assuming first channel (e.g., MLII)

        # Calculate the "time_ms" column based on the sampling frequency
        sampling_frequency = header.fs
        num_samples = len(record.p_signal)
        time_ms = [1000 * i / sampling_frequency for i in range(num_samples)]

        # Create a DataFrame with signal data, "Sample index," and "Time (ms)"
        ecg_data = pd.DataFrame({'Sample index': range(num_samples), 'Time (ms)': time_ms})
        for i in range(header.n_sig):
            ecg_data[header.sig_name[i]] = record.p_signal[:, i]

        print(f"Columns in the ecg_data DataFrame for record {record_name}: {ecg_data.columns}")

        # Preprocess the ECG signals
        ecg_data = preprocess_ecg_signal(ecg_data)

        # Read the annotations
        annotations = wfdb.rdann(os.path.join(data_dir, record_name), 'atr')
        print(f"Annotation columns for record {record_name}: {annotations.__dict__}")

        # Extract annotation sample indices and symbols
        ann_sample_indices = annotations.sample
        ann_symbols = annotations.symbol

        # Initialize annotation and description columns
        ecg_data['Symbol'] = ''
        ecg_data['Description'] = ''

        # Populate 'Symbol' and 'Description' only for recognized symbols
        for idx, symbol in zip(ann_sample_indices, ann_symbols):
            if symbol in annotation_equiv:
                new_symbol = annotation_equiv[symbol]
                ecg_data.at[idx, 'Symbol'] = new_symbol
                ecg_data.at[idx, 'Description'] = annotation_desc[new_symbol]

        # Add the 'Channels' column
        ecg_data['Channels'] = ', '.join(header.sig_name)

        # Build annotation details after preprocessing
        annotation_details = []
        for idx, symbol in zip(ann_sample_indices, ann_symbols):
            if symbol in annotation_equiv:
                new_symbol = annotation_equiv[symbol]
                annotation_info = {
                    'Sample index': idx,
                    'Time (ms)': time_ms[idx],
                    'Symbol': new_symbol,
                    'Description': annotation_desc[new_symbol],
                    'Channels': ', '.join(header.sig_name)
                }
                # Include the processed signals for each channel
                for sig_name in header.sig_name:
                    annotation_info[sig_name] = ecg_data.at[idx, sig_name]
                annotation_details.append(annotation_info)

        annotation_details_df = pd.DataFrame(annotation_details)

        # Save main DataFrame as a CSV file
        csv_filename = os.path.join(output_dir, f'{record_name}.csv')
        ecg_data.to_csv(csv_filename, index=False)
        print(f'Record {record_name} saved as {csv_filename}')

        # Save annotation details to another CSV file
        annotation_csv_filename = os.path.join(output_dir, f'{record_name}_annotation_details.csv')
        print(f"Saving annotation details to {annotation_csv_filename} with the first 5 rows:\n{annotation_details_df.head()}")
        annotation_details_df.to_csv(annotation_csv_filename, index=False)

        # Extract beats
        beat_segments = extract_beats(record, annotations, signal, time_ms)

        # Build a DataFrame for extracted beats
        max_len = max(len(seg[0]) for seg in beat_segments) if beat_segments else 0
        beat_data = {
            'Symbol': [annotation_equiv[seg[2]] for seg in beat_segments]
        }
        for i in range(max_len):
            beat_data[f'Sample_{i}'] = [seg[0][i] if i < len(seg[0]) else np.nan for seg in beat_segments]
            beat_data[f'Time_ms_{i}'] = [seg[1][i] if i < len(seg[1]) else np.nan for seg in beat_segments]

        beats_df = pd.DataFrame(beat_data)
        beats_csv_filename = os.path.join(output_dir, f'{record_name}_beats.csv')
        beats_df.to_csv(beats_csv_filename, index=False)
        print(f'Extracted beats for record {record_name} and saved to {beats_csv_filename}')

    except FileNotFoundError:
        print(f'Record {record_name} not found. Skipping...')

# Process each record in the specified range
for record_number in record_numbers:
    process_record(record_number)

print("Processing complete.")
