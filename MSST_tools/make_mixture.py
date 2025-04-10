import os
from pydub import AudioSegment

def mix_wav_files(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    for folder in subfolders:
        aspiration_path = os.path.join(folder, "aspiration.wav")
        other_path = os.path.join(folder, "other.wav")
        mixture_path = os.path.join(folder, "mixture.wav")

        if os.path.exists(aspiration_path) and os.path.exists(other_path):
            aspiration_audio = AudioSegment.from_wav(aspiration_path)
            other_audio = AudioSegment.from_wav(other_path)
            mixture_audio = aspiration_audio.overlay(other_audio)

            mixture_audio.export(mixture_path, format="wav")
            print(f"混音完成: {mixture_path}")
        else:
            print(f"缺少文件，跳过文件夹: {folder}")

top_folder_path = './validation'
mix_wav_files(top_folder_path)
