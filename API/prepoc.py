import librosa
import numpy as np
from scipy.signal import butter, filtfilt, medfilt
from scipy import signal
import pywt
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

def extract_features(audio_file_path):
    # Load heart sound audio file
    print(audio_file_path)
    audio_data, sample_rate = librosa.load(audio_file_path, sr=None, mono=True)
    # Select the first 5 seconds of the audio
    audio_data_5s = audio_data[:5 * sample_rate]
    # Normalize the heart sound signal
    audio_data_norm = (audio_data_5s - audio_data_5s.min()) / (audio_data_5s.max() - audio_data_5s.min())
    # Define the filter parameters
    fs = sample_rate  # Sample rate
    lowcut = 25  # Low-pass cut-off frequency in Hz
    highcut = 400  # High-pass cut-off frequency in Hz
    order = 2  # Filter order
    # Compute the Nyquist frequency
    nyq = 0.5 * fs
    # Compute the filter coefficients
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    # Apply the filter to the audio data
    audio_data_filtered = filtfilt(b, a, audio_data_norm)
    # Apply median filter to remove spikes
    audio_data_filtered_median = medfilt(audio_data_filtered, kernel_size=3)
    # Define window size and overlap
    window_size = 0.1 # in seconds
    overlap = 0.05 # in seconds
    # Calculate number of samples per window and overlap
    samples_per_window = int(window_size * sample_rate)
    samples_per_overlap = int(overlap * sample_rate)
    # Calculate number of windows
    num_windows = int(np.ceil(len(audio_data_filtered_median) / samples_per_overlap))
    # Initialize feature matrix
    feature_matrix = np.zeros((num_windows, 4))
    # Extract features from each window
    for i in range(num_windows):
        # Calculate start and end indices for current window
        start_index = i * samples_per_overlap
        end_index = start_index + samples_per_window
        # Extract filtered data for current window
        window_data = audio_data_filtered_median[start_index:end_index]
        # Remove spikes
        window_data = signal.medfilt(window_data, kernel_size=5)
        # Extract Hilbert envelope
        hilbert_env = np.abs(signal.hilbert(window_data))
        feature_matrix[i, 0] = np.mean(hilbert_env)
        # Extract Homomorphic envelope
        homomorphic_env = np.abs(signal.hilbert(np.log(np.abs(window_data))))
        feature_matrix[i, 1] = np.mean(homomorphic_env)
        # Extract Wavelet envelope
        coeffs = pywt.wavedec(window_data, wavelet='db4', level=4)
        cA4, cD4, cD3, cD2, cD1 = coeffs
        wavelet_env = pywt.waverec([cA4, cD4, cD3, cD2, cD1], wavelet='db4')
        feature_matrix[i, 2] = np.mean(wavelet_env)
        # Extract Power spectral density envelope
        f, Pxx = signal.welch(window_data, fs=sample_rate, nperseg=samples_per_window)
        psd_env = np.sqrt(Pxx)
        feature_matrix[i, 3] = np.mean(psd_env)
    # Create dataframe with features
    feature_names = ['hilbert_env', 'homomorphic_env', 'wavelet_env', 'psd_env']
    df_features = pd.DataFrame(data=feature_matrix, columns=feature_names)
    df_features['start_time'] = np.arange(0, num_windows * samples_per_overlap, samples_per_overlap) / sample_rate
    df_features['end_time'] = df_features['start_time'] + window_size
    df_features = df_features.drop(['start_time', 'end_time'], axis=1)
    lst_columns_1= []
    for i in range(100):
        for column in df_features.columns:
            lst_columns_1.append(f'Decomp{i}_t_{column}')
    new_data_1= pd.DataFrame(df_features.to_numpy().reshape(1,-1), columns=lst_columns_1)
    new_data_1 = new_data_1.fillna(0)
    
     #Wavelet features
    # wavelet transform
    wavelet = pywt.Wavelet('db4')
    coefficients = pywt.wavedec(audio_data, wavelet, level=4)
    # extracting features
    features_M = []
    for c in coefficients:
        duration = len(c) / sample_rate  
        energy = np.sum(np.square(c))
        peak_amplitude = np.max(np.abs(c))
        mean_amplitude = np.mean(np.abs(c)) 
        std_amplitude = np.std(np.abs(c))
        skew_amplitude = np.abs(np.mean(c**3)) / np.power(np.mean(c**2), 3/2)
        kurtosis_amplitude = np.mean(c**4) / np.power(np.mean(c**2), 2)
        stats = {'Duration': duration, 'Energy': energy, 'Peak Amplitude': peak_amplitude,
                 'Mean Amplitude': mean_amplitude, 'Std Amplitude': std_amplitude,
                 'Skew Amplitude': skew_amplitude, 'Kurtosis Amplitude': kurtosis_amplitude}
        features_M.append(stats)
        final_dfs_M = pd.DataFrame(features_M)
    lst_columns_2 =[]
    for i in range(5):
        for column in final_dfs_M.columns:
            lst_columns_2.append(f'Decomp{i}_{column}')
    new_data_2 = pd.DataFrame(final_dfs_M.to_numpy().reshape(1,-1), columns=lst_columns_2)
    
    # Concatenate the two data frames
    merged_features = pd.concat([new_data_2, new_data_1], axis=1)
    
    #Normalizing the features
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    normalized_df = pd.DataFrame(scaler.transform(merged_features))
    
    return normalized_df

    