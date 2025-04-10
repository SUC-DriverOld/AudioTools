import librosa
import os
import random
import numpy as np
import soundfile as sf
import traceback
from tqdm import tqdm


def to_mp3(input_file):
    bps = random.choice(["64k", "96k", "128k", "192k", "256k", "320k"])
    to_mp3 = f"ffmpeg -i \"{input_file}\" -ar 44100 -ac 2 -b:a {bps} temp.mp3 -y"
    os.system(f"{to_mp3} > {os.devnull} 2>&1")
    audio, _ = librosa.load("temp.mp3", sr=44100, mono=False)
    os.remove("temp.mp3")
    return audio

def match_length(audio1, audio2):
    if audio1.shape[-1] != audio2.shape[-1]:
        min_length = min(audio1.shape[-1], audio2.shape[-1])
        audio1 = audio1[..., :min_length]
        audio2 = audio2[..., :min_length]
    return audio1, audio2


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", type=str, default="input")
    parser.add_argument("-o", "--output_folder", type=str, default="output")
    args = parser.parse_args()

    index = 1
    for audio_name in tqdm(os.listdir(args.input_folder)):
        try:
            wav_audio, _ = librosa.load(os.path.join(args.input_folder, audio_name), sr=44100, mono=False)
            if len(wav_audio.shape) == 1:
                wav_audio = np.stack([wav_audio, wav_audio], axis=0)
                print(f"Converted {audio_name} to stereo.")
            mp3_audio = to_mp3(os.path.join(args.input_folder, audio_name))
            wav_audio, mp3_audio = match_length(wav_audio, mp3_audio)
            addition = mp3_audio - wav_audio

            store_dir = os.path.join(args.output_folder, str(index))
            os.makedirs(store_dir, exist_ok=True)
            restored = sf.write(os.path.join(store_dir, "restored.wav"), wav_audio.T, 44100)
            addition = sf.write(os.path.join(store_dir, "addition.wav"), addition.T, 44100)
            mixture = sf.write(os.path.join(store_dir, "mixture.wav"), mp3_audio.T, 44100)

        except Exception as e:
            print(f"Cound not process {audio_name}. Error: {str(e)}")
            traceback.print_exc()
            continue

        index += 1
