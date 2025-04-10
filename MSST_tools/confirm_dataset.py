import os
import argparse
import wave

def check_audio_files(directory):
    subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    missing_folders = []
    non_stereo_files = []

    for folder in subfolders:
        files = os.listdir(folder)
        # Check for missing files
        if 'other.wav' not in files or 'aspiration.wav' not in files:
            missing_folders.append(folder)
        else:
            # Check if the files are stereo
            other_path = os.path.join(folder, 'other.wav')
            aspiration_path = os.path.join(folder, 'aspiration.wav')
            
            if not is_stereo(other_path):
                non_stereo_files.append(other_path)
            if not is_stereo(aspiration_path):
                non_stereo_files.append(aspiration_path)

    if missing_folders:
        print("The following subfolders are missing either 'other.wav' or 'aspiration.wav':")
        for folder in missing_folders:
            print(folder)
    else:
        print("All subfolders have both 'other.wav' and 'aspiration.wav'.")

    if non_stereo_files:
        print("The following files are not stereo:")
        for file in non_stereo_files:
            print(file)
    else:
        print("All files are stereo.")

def is_stereo(file_path):
    """Check if the audio file is stereo."""
    try:
        with wave.open(file_path, 'rb') as wf:
            return wf.getnchannels() == 2  # Stereo files have 2 channels
    except wave.Error as e:
        print(f"Error reading {file_path}: {e}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='input', help='Input directory')
    args = parser.parse_args()
    check_audio_files(args.input_dir)
    print("Finish!")
