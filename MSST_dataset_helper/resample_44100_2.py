import os
from pydub import AudioSegment

def resample_audio_in_place(folder):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)

                print(f"正在处理: {file_path}")
                try:
                    audio = AudioSegment.from_file(file_path)
                    audio = audio.set_frame_rate(44100).set_channels(2)

                    audio.export(file_path, format="wav")
                    print(f"已覆盖: {file_path}")
                except Exception as e:
                    print(f"处理失败: {file_path}，错误: {e}")

if __name__ == "__main__":
    folder = r"valid_raw\male"
    resample_audio_in_place(folder)
