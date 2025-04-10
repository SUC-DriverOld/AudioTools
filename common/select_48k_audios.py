import os
import shutil
import librosa
import numpy as np
import scipy.signal
import argparse
import multiprocessing
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Filter Audio Files")
parser.add_argument("-i", "--input_folder", type=str, default="your_input_folder", help="Input folder containing audio files")
parser.add_argument("-o", "--output_folder", type=str, default="filtered_out", help="Output folder for filtered audio files")
parser.add_argument("--threads", type=int, default=4, help="Number of threads to use for processing")
parser.add_argument("--threshold", type=float, default=1.0e-6, help="Threshold for high-frequency energy")
args = parser.parse_args()

input_folder = args.input_folder
output_folder = args.output_folder
threads = args.threads
threshold = args.threshold

def has_valid_high_freq(audio, sr=48000, freq_range=(22050, 24000), threshold=threshold):
    f, t, Sxx = scipy.signal.spectrogram(audio, sr, nperseg=2048, nfft=2048, mode='magnitude')
    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    mean_energy = np.mean(Sxx[freq_mask])
    return mean_energy > threshold

def process(audio_file):
    try:
        audio, sr = librosa.load(audio_file, sr=48000, mono=True)
        if not has_valid_high_freq(audio, sr=48000):
            print(f"Moving {audio_file} (Lacks High-Frequency Info)")
            shutil.move(audio_file, output_folder)
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")

def filter_audio():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    audio_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.aac')):
                audio_files.append(os.path.join(root, file))

    p = multiprocessing.Pool(threads)
    for _ in tqdm(p.imap_unordered(process, audio_files), total=len(audio_files)):
        pass

if __name__ == "__main__":
    filter_audio()
