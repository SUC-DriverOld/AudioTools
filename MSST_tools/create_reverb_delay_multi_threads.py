import os
import argparse
import librosa
import numpy as np
import soundfile as sf
from pedalboard import Pedalboard, Reverb, Delay, HighpassFilter, LowpassFilter
from random import uniform
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def random_effect(audio, sr):
    reverb = Pedalboard([
        Delay(
            delay_seconds=uniform(0.001, 0.100), # 1-100ms
            feedback=0.0, # no feedback
            mix=1.0 # full wet
        ), # pre-delay
        Reverb(
            room_size=uniform(0.1, 8.0), # 0.1-8.0
            damping=uniform(0.5, 1.0), # 0.5-1.0
            wet_level=1.0, # full wet
            dry_level=0.0, # no dry signal
            width=uniform(0.5, 1.0) # 0.5-1.0
        ), # reverb
        HighpassFilter(cutoff_frequency_hz=uniform(100, 800)), # 100-800Hz, 12db low cut
        LowpassFilter(cutoff_frequency_hz=uniform(4000, 15000)) # 4-15kHz, 12db high cut
    ])

    effect = uniform(0.2, 0.5) * reverb(audio, sr) # 20-50% effect
    mix = effect + audio # mixture of effect and original audio

    return mix, effect

def process_file(file, input_folder, output_folder, index, sr):
    try:
        audio, _ = librosa.load(os.path.join(input_folder, file), sr=sr)
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio], axis=1)
        effect = random_effect(audio.T, sr)
    except Exception as e:
        print(f"Failed to process file: {file}. Error: {e}")
        return False

    output_path = os.path.join(output_folder, str(index))
    os.makedirs(output_path, exist_ok=True)

    try:
        sf.write(os.path.join(output_path, "mixture.wav"), effect[0].T, sr, subtype='PCM_16')
        sf.write(os.path.join(output_path, "other.wav"), effect[1].T, sr, subtype='PCM_16')
        sf.write(os.path.join(output_path, "dry.wav"), audio, sr, subtype='PCM_16')
        os.remove(os.path.join(input_folder, file))
    except Exception as e:
        print(f"Failed to save file for {file}. Error: {e}")
        return False

    return True

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Add random reverb and delay effects to audio files using multithreading.')
    argparser.add_argument('-i', '--input_folder', type=str, default="trainset", help='Path to the input folder containing audio files.')
    argparser.add_argument('-o', '--output_folder', type=str, default="train2", help='Path to the output folder for processed audio files.')
    argparser.add_argument('-t', '--threads', type=int, default=32, help='Number of threads to use for processing.')
    args = argparser.parse_args()

    sr = 44100
    input_files = os.listdir(args.input_folder)

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(process_file, file, args.input_folder, args.output_folder, index, sr): file for index, file in enumerate(input_files, start=1)}
        for future in tqdm(futures, total=len(input_files)):
            future.result()