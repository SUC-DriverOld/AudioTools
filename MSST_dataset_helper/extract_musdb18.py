import os
import stempeg
import soundfile as sf


def split_stems(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".mp4"):
            input_path = os.path.join(input_folder, file_name)

            base_name = os.path.splitext(file_name)[0]
            output_subfolder = os.path.join(output_folder, base_name)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)

            print(f"正在处理 {file_name}...")

            stems, rate = stempeg.read_stems(input_path)
            track_names = ["mixture.wav", "drums.wav", "bass.wav", "other.wav", "vocals.wav"]
            for i, track_name in enumerate(track_names):
                output_path = os.path.join(output_subfolder, track_name)
                sf.write(output_path, stems[i], rate)

    print("音轨拆分完成！")

input_folder = r"musdb18\test"
output_folder = r"musdb18\extract_test"

split_stems(input_folder, output_folder)
