import librosa
import random
import os
import numpy as np
import soundfile as sf
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def match_length(audio1, audio2):
    if audio1.shape[-1] != audio2.shape[-1]:
        if audio1.shape[-1] < audio2.shape[-1]:
            audio2_start = random.randint(0, audio2.shape[-1] - audio1.shape[-1])
            audio2 = audio2[..., audio2_start:audio2_start + audio1.shape[-1]]
        else:
            audio1_start = random.randint(0, audio1.shape[-1] - audio2.shape[-1])
            audio1 = audio1[..., audio1_start:audio1_start + audio2.shape[-1]]
    return audio1, audio2

def process_audio_pair(i, input_dir, output_dir, all_files):
    audio1 = random.choice(all_files)
    audio2 = random.choice(all_files)
    audio1, _ = librosa.load(os.path.join(input_dir, audio1), sr=44100, mono=False)
    audio2, _ = librosa.load(os.path.join(input_dir, audio2), sr=44100, mono=False)
    
    if len(audio1.shape) == 1:
        audio1 = np.stack([audio1, audio1], axis=0)
    if len(audio2.shape) == 1:
        audio2 = np.stack([audio2, audio2], axis=0)
    
    audio1, audio2 = match_length(audio1, audio2)
    mixture = audio1 + audio2
    
    store_dir = os.path.join(output_dir, str(i))
    os.makedirs(store_dir, exist_ok=True)
    
    sf.write(os.path.join(store_dir, "mixture.wav"), mixture.T, 44100)
    sf.write(os.path.join(store_dir, "vocal_1.wav"), audio1.T, 44100)
    sf.write(os.path.join(store_dir, "vocal_2.wav"), audio2.T, 44100)

def create_multi_speaker_dataset(input_dir, output_dir, total=1000, num_threads=8):
    all_files = [audio for audio in os.listdir(input_dir) if audio.lower().endswith(('.wav', 'flac', 'mp3'))]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(process_audio_pair, i, input_dir, output_dir, all_files): i for i in range(total)}
        with tqdm.tqdm(total=total) as pbar:
            for _ in as_completed(futures):
                pbar.update(1)

if __name__ == "__main__":
    create_multi_speaker_dataset("data/audios", "data/multi_speaker_audios", total=1000, num_threads=8)
