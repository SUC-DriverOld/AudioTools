import os
from pydub import AudioSegment
import argparse


def merge_audio_files(input_folder, output_file):
    combined_audio = AudioSegment.empty()

    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(('.mp3', '.wav', '.ogg', '.flac')):
            file_path = os.path.join(input_folder, filename)
            audio = AudioSegment.from_file(file_path)
            combined_audio += audio

    combined_audio.export(output_file, format="wav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="merge audio files")
    parser.add_argument("-i", "--input_floder", required=True,
                        help="input file folder path")
    parser.add_argument("-o", "--output_file", required=False,
                        default="output.wav", help="output file name (include suffix)")
    args = parser.parse_args()
    merge_audio_files(args.input_floder, args.output_file)
    print(f"Merged audio files saved to {args.output_file}")
