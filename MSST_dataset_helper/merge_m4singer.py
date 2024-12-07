import os
from pydub import AudioSegment

def merge_wav_files(input_folder, output_folder):
    for subdir, _, files in os.walk(input_folder):
        if subdir == input_folder:
            continue

        subfolder_name = os.path.basename(subdir)
        wav_files = [f for f in files if f.endswith('.wav')]
        if not wav_files:
            continue

        combined_audio = AudioSegment.empty()
        for wav_file in wav_files:
            wav_path = os.path.join(subdir, wav_file)
            audio = AudioSegment.from_wav(wav_path)
            combined_audio += audio

        output_wav_path = os.path.join(output_folder, f'{subfolder_name}.wav')
        combined_audio.export(output_wav_path, format='wav')
        
        print(f"合并完成：{output_wav_path}")

if __name__ == "__main__":
    input_folder = "m4singer"
    output_folder = "m4singer_output"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    merge_wav_files(input_folder, output_folder)
