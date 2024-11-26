import os
import argparse
import librosa
import numpy as np
import soundfile as sf
from pedalboard import Pedalboard, Reverb, Delay, HighpassFilter, LowpassFilter
from random import uniform
from tqdm import tqdm


def random_effect(audio, sr):
    reverb = Pedalboard([
        Delay(
            delay_seconds=uniform(0.001, 0.100), 
            feedback=0.0, 
            mix=1.0
        ),
        Reverb(
            room_size=uniform(0.1, 0.8),
            damping=uniform(0.1, 0.8),
            wet_level=1.0, 
            dry_level=0.0, 
            width=uniform(0.6, 1.0)
        ),
        HighpassFilter(cutoff_frequency_hz=uniform(100, 1000)),
        LowpassFilter(cutoff_frequency_hz=uniform(4000, 12000))
    ])

    delay = Pedalboard([
        Delay(
            delay_seconds=uniform(0.05, 0.500),
            feedback=uniform(0.1, 0.5),
            mix=1.0
        ),
        Reverb(
            room_size=uniform(0.05, 0.3),
            damping=uniform(0.1, 0.8),
            wet_level=0.2,
            dry_level=0.8,
            width=uniform(0.6, 1.0)
        ),
        HighpassFilter(cutoff_frequency_hz=uniform(100, 1000)),
        LowpassFilter(cutoff_frequency_hz=uniform(3000, 10000))
    ])

    effect = uniform(0.1, 0.4) * reverb(audio, sr) + uniform(0.1, 0.4) * delay(audio, sr)
    mix = effect + audio

    return mix, effect


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Add random reverb and delay effects to an audio file.')
    argparser.add_argument('-i', '--input_folder', type=str, default="train", help='Path to the input audio file.')
    argparser.add_argument('-o', '--output_folder', type=str, default="dataset_train", help='Path to the output audio file.')
    args = argparser.parse_args()

    index = 1
    sr = 44100
    for file in tqdm(os.listdir(args.input_folder)):
        try:
            audio, _ = librosa.load(os.path.join(args.input_folder, file), sr=sr)
            if len(audio.shape) == 1:
                audio = np.stack([audio, audio], axis=1)
            effect = random_effect(audio.T, sr)
        except:
            print(f"Failed to process file: {file}")
            continue

        os.makedirs(os.path.join(args.output_folder, str(index)), exist_ok=True)

        sf.write(os.path.join(args.output_folder, str(index), "mixture.wav"), effect[0].T, sr, subtype='PCM_16')
        sf.write(os.path.join(args.output_folder, str(index), "other.wav"), effect[1].T, sr, subtype='PCM_16')
        sf.write(os.path.join(args.output_folder, str(index), "dry.wav"), audio, sr, subtype='PCM_16')

        index += 1
