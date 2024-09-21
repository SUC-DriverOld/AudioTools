import os
import shutil
import argparse

def organize_audio_files(source_dir, output_dir):
    wav_files = [f for f in os.listdir(source_dir) if f.endswith('.wav')]
    groups = {}
    for file in wav_files:
        base_name = file.rsplit('_Asp', 1)[0].rsplit('.wav', 1)[0]
        if base_name not in groups:
            groups[base_name] = []
        groups[base_name].append(file)

    folder_index = 1
    for base_name, files in groups.items():

        target_folder = os.path.join(output_dir, str(folder_index))
        os.makedirs(target_folder, exist_ok=True)

        for file in files:
            source_path = os.path.join(source_dir, file)
            if '_Asp' in file:
                target_path = os.path.join(target_folder, 'aspiration.wav')
            else:
                target_path = os.path.join(target_folder, 'other.wav')

            shutil.copy(source_path, target_path)

        folder_index += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='input', help='Input directory')
    parser.add_argument('-o', '--output_dir', type=str, default='output', help='Output directory')
    args = parser.parse_args()
    organize_audio_files(args.input_dir, args.output_dir)
    print("Finish!")
