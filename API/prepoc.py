import librosa
import numpy as np
from scipy.signal import butter, filtfilt, medfilt
from scipy import signal
import pywt
import pandas as pd

def dummy_prepoc():
    return pd.read_csv('final_data.csv', nrows = 1).drop(columns=['Unnamed: 0', 'wav_file', 'Class (-1=normal 1=abnormal)'])

def extract_features(audio_file_path):
    # Load heart sound audio file
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
    merged_features = pd.concat([new_data_1, new_data_2], axis=1)
    
    return merged_features[['Decomp0_Duration',
 'Decomp0_Energy',
 'Decomp0_Peak Amplitude',
 'Decomp0_Mean Amplitude',
 'Decomp0_Std Amplitude',
 'Decomp0_Skew Amplitude',
 'Decomp0_Kurtosis Amplitude',
 'Decomp1_Duration',
 'Decomp1_Energy',
 'Decomp1_Peak Amplitude',
 'Decomp1_Mean Amplitude',
 'Decomp1_Std Amplitude',
 'Decomp1_Skew Amplitude',
 'Decomp1_Kurtosis Amplitude',
 'Decomp2_Duration',
 'Decomp2_Energy',
 'Decomp2_Peak Amplitude',
 'Decomp2_Mean Amplitude',
 'Decomp2_Std Amplitude',
 'Decomp2_Skew Amplitude',
 'Decomp2_Kurtosis Amplitude',
 'Decomp3_Duration',
 'Decomp3_Energy',
 'Decomp3_Peak Amplitude',
 'Decomp3_Mean Amplitude',
 'Decomp3_Std Amplitude',
 'Decomp3_Skew Amplitude',
 'Decomp3_Kurtosis Amplitude',
 'Decomp4_Duration',
 'Decomp4_Energy',
 'Decomp4_Peak Amplitude',
 'Decomp4_Mean Amplitude',
 'Decomp4_Std Amplitude',
 'Decomp4_Skew Amplitude',
 'Decomp4_Kurtosis Amplitude',
 'Decomp0_t_hilbert_env',
 'Decomp0_t_homomorphic_env',
 'Decomp0_t_wavelet_env',
 'Decomp0_t_psd_env',
 'Decomp1_t_hilbert_env',
 'Decomp1_t_homomorphic_env',
 'Decomp1_t_wavelet_env',
 'Decomp1_t_psd_env',
 'Decomp2_t_hilbert_env',
 'Decomp2_t_homomorphic_env',
 'Decomp2_t_wavelet_env',
 'Decomp2_t_psd_env',
 'Decomp3_t_hilbert_env',
 'Decomp3_t_homomorphic_env',
 'Decomp3_t_wavelet_env',
 'Decomp3_t_psd_env',
 'Decomp4_t_hilbert_env',
 'Decomp4_t_homomorphic_env',
 'Decomp4_t_wavelet_env',
 'Decomp4_t_psd_env',
 'Decomp5_t_hilbert_env',
 'Decomp5_t_homomorphic_env',
 'Decomp5_t_wavelet_env',
 'Decomp5_t_psd_env',
 'Decomp6_t_hilbert_env',
 'Decomp6_t_homomorphic_env',
 'Decomp6_t_wavelet_env',
 'Decomp6_t_psd_env',
 'Decomp7_t_hilbert_env',
 'Decomp7_t_homomorphic_env',
 'Decomp7_t_wavelet_env',
 'Decomp7_t_psd_env',
 'Decomp8_t_hilbert_env',
 'Decomp8_t_homomorphic_env',
 'Decomp8_t_wavelet_env',
 'Decomp8_t_psd_env',
 'Decomp9_t_hilbert_env',
 'Decomp9_t_homomorphic_env',
 'Decomp9_t_wavelet_env',
 'Decomp9_t_psd_env',
 'Decomp10_t_hilbert_env',
 'Decomp10_t_homomorphic_env',
 'Decomp10_t_wavelet_env',
 'Decomp10_t_psd_env',
 'Decomp11_t_hilbert_env',
 'Decomp11_t_homomorphic_env',
 'Decomp11_t_wavelet_env',
 'Decomp11_t_psd_env',
 'Decomp12_t_hilbert_env',
 'Decomp12_t_homomorphic_env',
 'Decomp12_t_wavelet_env',
 'Decomp12_t_psd_env',
 'Decomp13_t_hilbert_env',
 'Decomp13_t_homomorphic_env',
 'Decomp13_t_wavelet_env',
 'Decomp13_t_psd_env',
 'Decomp14_t_hilbert_env',
 'Decomp14_t_homomorphic_env',
 'Decomp14_t_wavelet_env',
 'Decomp14_t_psd_env',
 'Decomp15_t_hilbert_env',
 'Decomp15_t_homomorphic_env',
 'Decomp15_t_wavelet_env',
 'Decomp15_t_psd_env',
 'Decomp16_t_hilbert_env',
 'Decomp16_t_homomorphic_env',
 'Decomp16_t_wavelet_env',
 'Decomp16_t_psd_env',
 'Decomp17_t_hilbert_env',
 'Decomp17_t_homomorphic_env',
 'Decomp17_t_wavelet_env',
 'Decomp17_t_psd_env',
 'Decomp18_t_hilbert_env',
 'Decomp18_t_homomorphic_env',
 'Decomp18_t_wavelet_env',
 'Decomp18_t_psd_env',
 'Decomp19_t_hilbert_env',
 'Decomp19_t_homomorphic_env',
 'Decomp19_t_wavelet_env',
 'Decomp19_t_psd_env',
 'Decomp20_t_hilbert_env',
 'Decomp20_t_homomorphic_env',
 'Decomp20_t_wavelet_env',
 'Decomp20_t_psd_env',
 'Decomp21_t_hilbert_env',
 'Decomp21_t_homomorphic_env',
 'Decomp21_t_wavelet_env',
 'Decomp21_t_psd_env',
 'Decomp22_t_hilbert_env',
 'Decomp22_t_homomorphic_env',
 'Decomp22_t_wavelet_env',
 'Decomp22_t_psd_env',
 'Decomp23_t_hilbert_env',
 'Decomp23_t_homomorphic_env',
 'Decomp23_t_wavelet_env',
 'Decomp23_t_psd_env',
 'Decomp24_t_hilbert_env',
 'Decomp24_t_homomorphic_env',
 'Decomp24_t_wavelet_env',
 'Decomp24_t_psd_env',
 'Decomp25_t_hilbert_env',
 'Decomp25_t_homomorphic_env',
 'Decomp25_t_wavelet_env',
 'Decomp25_t_psd_env',
 'Decomp26_t_hilbert_env',
 'Decomp26_t_homomorphic_env',
 'Decomp26_t_wavelet_env',
 'Decomp26_t_psd_env',
 'Decomp27_t_hilbert_env',
 'Decomp27_t_homomorphic_env',
 'Decomp27_t_wavelet_env',
 'Decomp27_t_psd_env',
 'Decomp28_t_hilbert_env',
 'Decomp28_t_homomorphic_env',
 'Decomp28_t_wavelet_env',
 'Decomp28_t_psd_env',
 'Decomp29_t_hilbert_env',
 'Decomp29_t_homomorphic_env',
 'Decomp29_t_wavelet_env',
 'Decomp29_t_psd_env',
 'Decomp30_t_hilbert_env',
 'Decomp30_t_homomorphic_env',
 'Decomp30_t_wavelet_env',
 'Decomp30_t_psd_env',
 'Decomp31_t_hilbert_env',
 'Decomp31_t_homomorphic_env',
 'Decomp31_t_wavelet_env',
 'Decomp31_t_psd_env',
 'Decomp32_t_hilbert_env',
 'Decomp32_t_homomorphic_env',
 'Decomp32_t_wavelet_env',
 'Decomp32_t_psd_env',
 'Decomp33_t_hilbert_env',
 'Decomp33_t_homomorphic_env',
 'Decomp33_t_wavelet_env',
 'Decomp33_t_psd_env',
 'Decomp34_t_hilbert_env',
 'Decomp34_t_homomorphic_env',
 'Decomp34_t_wavelet_env',
 'Decomp34_t_psd_env',
 'Decomp35_t_hilbert_env',
 'Decomp35_t_homomorphic_env',
 'Decomp35_t_wavelet_env',
 'Decomp35_t_psd_env',
 'Decomp36_t_hilbert_env',
 'Decomp36_t_homomorphic_env',
 'Decomp36_t_wavelet_env',
 'Decomp36_t_psd_env',
 'Decomp37_t_hilbert_env',
 'Decomp37_t_homomorphic_env',
 'Decomp37_t_wavelet_env',
 'Decomp37_t_psd_env',
 'Decomp38_t_hilbert_env',
 'Decomp38_t_homomorphic_env',
 'Decomp38_t_wavelet_env',
 'Decomp38_t_psd_env',
 'Decomp39_t_hilbert_env',
 'Decomp39_t_homomorphic_env',
 'Decomp39_t_wavelet_env',
 'Decomp39_t_psd_env',
 'Decomp40_t_hilbert_env',
 'Decomp40_t_homomorphic_env',
 'Decomp40_t_wavelet_env',
 'Decomp40_t_psd_env',
 'Decomp41_t_hilbert_env',
 'Decomp41_t_homomorphic_env',
 'Decomp41_t_wavelet_env',
 'Decomp41_t_psd_env',
 'Decomp42_t_hilbert_env',
 'Decomp42_t_homomorphic_env',
 'Decomp42_t_wavelet_env',
 'Decomp42_t_psd_env',
 'Decomp43_t_hilbert_env',
 'Decomp43_t_homomorphic_env',
 'Decomp43_t_wavelet_env',
 'Decomp43_t_psd_env',
 'Decomp44_t_hilbert_env',
 'Decomp44_t_homomorphic_env',
 'Decomp44_t_wavelet_env',
 'Decomp44_t_psd_env',
 'Decomp45_t_hilbert_env',
 'Decomp45_t_homomorphic_env',
 'Decomp45_t_wavelet_env',
 'Decomp45_t_psd_env',
 'Decomp46_t_hilbert_env',
 'Decomp46_t_homomorphic_env',
 'Decomp46_t_wavelet_env',
 'Decomp46_t_psd_env',
 'Decomp47_t_hilbert_env',
 'Decomp47_t_homomorphic_env',
 'Decomp47_t_wavelet_env',
 'Decomp47_t_psd_env',
 'Decomp48_t_hilbert_env',
 'Decomp48_t_homomorphic_env',
 'Decomp48_t_wavelet_env',
 'Decomp48_t_psd_env',
 'Decomp49_t_hilbert_env',
 'Decomp49_t_homomorphic_env',
 'Decomp49_t_wavelet_env',
 'Decomp49_t_psd_env',
 'Decomp50_t_hilbert_env',
 'Decomp50_t_homomorphic_env',
 'Decomp50_t_wavelet_env',
 'Decomp50_t_psd_env',
 'Decomp51_t_hilbert_env',
 'Decomp51_t_homomorphic_env',
 'Decomp51_t_wavelet_env',
 'Decomp51_t_psd_env',
 'Decomp52_t_hilbert_env',
 'Decomp52_t_homomorphic_env',
 'Decomp52_t_wavelet_env',
 'Decomp52_t_psd_env',
 'Decomp53_t_hilbert_env',
 'Decomp53_t_homomorphic_env',
 'Decomp53_t_wavelet_env',
 'Decomp53_t_psd_env',
 'Decomp54_t_hilbert_env',
 'Decomp54_t_homomorphic_env',
 'Decomp54_t_wavelet_env',
 'Decomp54_t_psd_env',
 'Decomp55_t_hilbert_env',
 'Decomp55_t_homomorphic_env',
 'Decomp55_t_wavelet_env',
 'Decomp55_t_psd_env',
 'Decomp56_t_hilbert_env',
 'Decomp56_t_homomorphic_env',
 'Decomp56_t_wavelet_env',
 'Decomp56_t_psd_env',
 'Decomp57_t_hilbert_env',
 'Decomp57_t_homomorphic_env',
 'Decomp57_t_wavelet_env',
 'Decomp57_t_psd_env',
 'Decomp58_t_hilbert_env',
 'Decomp58_t_homomorphic_env',
 'Decomp58_t_wavelet_env',
 'Decomp58_t_psd_env',
 'Decomp59_t_hilbert_env',
 'Decomp59_t_homomorphic_env',
 'Decomp59_t_wavelet_env',
 'Decomp59_t_psd_env',
 'Decomp60_t_hilbert_env',
 'Decomp60_t_homomorphic_env',
 'Decomp60_t_wavelet_env',
 'Decomp60_t_psd_env',
 'Decomp61_t_hilbert_env',
 'Decomp61_t_homomorphic_env',
 'Decomp61_t_wavelet_env',
 'Decomp61_t_psd_env',
 'Decomp62_t_hilbert_env',
 'Decomp62_t_homomorphic_env',
 'Decomp62_t_wavelet_env',
 'Decomp62_t_psd_env',
 'Decomp63_t_hilbert_env',
 'Decomp63_t_homomorphic_env',
 'Decomp63_t_wavelet_env',
 'Decomp63_t_psd_env',
 'Decomp64_t_hilbert_env',
 'Decomp64_t_homomorphic_env',
 'Decomp64_t_wavelet_env',
 'Decomp64_t_psd_env',
 'Decomp65_t_hilbert_env',
 'Decomp65_t_homomorphic_env',
 'Decomp65_t_wavelet_env',
 'Decomp65_t_psd_env',
 'Decomp66_t_hilbert_env',
 'Decomp66_t_homomorphic_env',
 'Decomp66_t_wavelet_env',
 'Decomp66_t_psd_env',
 'Decomp67_t_hilbert_env',
 'Decomp67_t_homomorphic_env',
 'Decomp67_t_wavelet_env',
 'Decomp67_t_psd_env',
 'Decomp68_t_hilbert_env',
 'Decomp68_t_homomorphic_env',
 'Decomp68_t_wavelet_env',
 'Decomp68_t_psd_env',
 'Decomp69_t_hilbert_env',
 'Decomp69_t_homomorphic_env',
 'Decomp69_t_wavelet_env',
 'Decomp69_t_psd_env',
 'Decomp70_t_hilbert_env',
 'Decomp70_t_homomorphic_env',
 'Decomp70_t_wavelet_env',
 'Decomp70_t_psd_env',
 'Decomp71_t_hilbert_env',
 'Decomp71_t_homomorphic_env',
 'Decomp71_t_wavelet_env',
 'Decomp71_t_psd_env',
 'Decomp72_t_hilbert_env',
 'Decomp72_t_homomorphic_env',
 'Decomp72_t_wavelet_env',
 'Decomp72_t_psd_env',
 'Decomp73_t_hilbert_env',
 'Decomp73_t_homomorphic_env',
 'Decomp73_t_wavelet_env',
 'Decomp73_t_psd_env',
 'Decomp74_t_hilbert_env',
 'Decomp74_t_homomorphic_env',
 'Decomp74_t_wavelet_env',
 'Decomp74_t_psd_env',
 'Decomp75_t_hilbert_env',
 'Decomp75_t_homomorphic_env',
 'Decomp75_t_wavelet_env',
 'Decomp75_t_psd_env',
 'Decomp76_t_hilbert_env',
 'Decomp76_t_homomorphic_env',
 'Decomp76_t_wavelet_env',
 'Decomp76_t_psd_env',
 'Decomp77_t_hilbert_env',
 'Decomp77_t_homomorphic_env',
 'Decomp77_t_wavelet_env',
 'Decomp77_t_psd_env',
 'Decomp78_t_hilbert_env',
 'Decomp78_t_homomorphic_env',
 'Decomp78_t_wavelet_env',
 'Decomp78_t_psd_env',
 'Decomp79_t_hilbert_env',
 'Decomp79_t_homomorphic_env',
 'Decomp79_t_wavelet_env',
 'Decomp79_t_psd_env',
 'Decomp80_t_hilbert_env',
 'Decomp80_t_homomorphic_env',
 'Decomp80_t_wavelet_env',
 'Decomp80_t_psd_env',
 'Decomp81_t_hilbert_env',
 'Decomp81_t_homomorphic_env',
 'Decomp81_t_wavelet_env',
 'Decomp81_t_psd_env',
 'Decomp82_t_hilbert_env',
 'Decomp82_t_homomorphic_env',
 'Decomp82_t_wavelet_env',
 'Decomp82_t_psd_env',
 'Decomp83_t_hilbert_env',
 'Decomp83_t_homomorphic_env',
 'Decomp83_t_wavelet_env',
 'Decomp83_t_psd_env',
 'Decomp84_t_hilbert_env',
 'Decomp84_t_homomorphic_env',
 'Decomp84_t_wavelet_env',
 'Decomp84_t_psd_env',
 'Decomp85_t_hilbert_env',
 'Decomp85_t_homomorphic_env',
 'Decomp85_t_wavelet_env',
 'Decomp85_t_psd_env',
 'Decomp86_t_hilbert_env',
 'Decomp86_t_homomorphic_env',
 'Decomp86_t_wavelet_env',
 'Decomp86_t_psd_env',
 'Decomp87_t_hilbert_env',
 'Decomp87_t_homomorphic_env',
 'Decomp87_t_wavelet_env',
 'Decomp87_t_psd_env',
 'Decomp88_t_hilbert_env',
 'Decomp88_t_homomorphic_env',
 'Decomp88_t_wavelet_env',
 'Decomp88_t_psd_env',
 'Decomp89_t_hilbert_env',
 'Decomp89_t_homomorphic_env',
 'Decomp89_t_wavelet_env',
 'Decomp89_t_psd_env',
 'Decomp90_t_hilbert_env',
 'Decomp90_t_homomorphic_env',
 'Decomp90_t_wavelet_env',
 'Decomp90_t_psd_env',
 'Decomp91_t_hilbert_env',
 'Decomp91_t_homomorphic_env',
 'Decomp91_t_wavelet_env',
 'Decomp91_t_psd_env',
 'Decomp92_t_hilbert_env',
 'Decomp92_t_homomorphic_env',
 'Decomp92_t_wavelet_env',
 'Decomp92_t_psd_env',
 'Decomp93_t_hilbert_env',
 'Decomp93_t_homomorphic_env',
 'Decomp93_t_wavelet_env',
 'Decomp93_t_psd_env',
 'Decomp94_t_hilbert_env',
 'Decomp94_t_homomorphic_env',
 'Decomp94_t_wavelet_env',
 'Decomp94_t_psd_env',
 'Decomp95_t_hilbert_env',
 'Decomp95_t_homomorphic_env',
 'Decomp95_t_wavelet_env',
 'Decomp95_t_psd_env',
 'Decomp96_t_hilbert_env',
 'Decomp96_t_homomorphic_env',
 'Decomp96_t_wavelet_env',
 'Decomp96_t_psd_env',
 'Decomp97_t_hilbert_env',
 'Decomp97_t_homomorphic_env',
 'Decomp97_t_wavelet_env',
 'Decomp97_t_psd_env',
 'Decomp98_t_hilbert_env',
 'Decomp98_t_homomorphic_env',
 'Decomp98_t_wavelet_env',
 'Decomp98_t_psd_env',
 'Decomp99_t_hilbert_env',
 'Decomp99_t_homomorphic_env',
 'Decomp99_t_wavelet_env',
 'Decomp99_t_psd_env']]
    
    
if __name__=="__main__" : 
    features = dummy_prepoc()
    print(features)
   