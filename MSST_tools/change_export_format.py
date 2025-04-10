import os
import json
import argparse

def read_svp_file(file_path):
    with open(file_path, encoding="utf-8") as file:
        data = json.loads(file.read()[:-1], strict=False)
    return data

def change_export_format(args):
    svp_files = [f for f in os.listdir(args.input_folder) if f.endswith('.svp')]
    file_index = 100

    for file in svp_files:
        try:
            data = read_svp_file(os.path.join(args.input_folder, file))
        except json.JSONDecodeError as e:
            print(e)
            print(f"Error reading {file}. Skipping...")
            continue

        data['renderConfig']["destination"] = args.destination
        data['renderConfig']["numChannels"] = args.num_channels
        data['renderConfig']["aspirationFormat"] = args.aspiration_format
        data['renderConfig']["bitDepth"] = args.bit_depth
        data['renderConfig']["sampleRate"] = args.sample_rate
        data['renderConfig']["exportMixDown"] = args.export_mixdown
        data['renderConfig']["exportPitch"] = args.export_pitch

        if not args.disable_fileindex:
            data['renderConfig']["filename"] = file_index
            save_file = os.path.join(args.input_folder, f"{file_index}.svp")
            os.unlink(os.path.join(args.input_folder, file))
        else:
            save_file = os.path.join(args.input_folder, file)

        with open(os.path.join(args.input_folder, file), 'w') as file:
            json.dump(data, file, indent=None)

        file_index += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=str, default='input', help='.svp project input folder')
    parser.add_argument('-d', '--destination', type=str, default='output', help='Music output destination')
    parser.add_argument('-nc', '--num_channels', choices=[1, 2], type=int, default=2, help='Number of channels')
    parser.add_argument('-af', '--aspiration_format', choices=['noAspiration', 'asChannels', 'asFiles', 'asIsolatedChannels', 'asIsolatedFiles'], type=str, default='asIsolatedFiles', help='Aspiration format')
    parser.add_argument('-bd', '--bit_depth', choices=[16, 24, 32], type=int, default=16, help='Bit depth')
    parser.add_argument('-sr', '--sample_rate', choices=[44100, 48000, 96000], type=int, default=44100, help='Sample rate')
    parser.add_argument('-em', '--export_mixdown', action='store_true', help='Export mixdown')
    parser.add_argument('-ep', '--export_pitch', action='store_true', help='Export pitch')
    parser.add_argument('-df', '--disable_fileindex', action='store_true', help='Disable file index')

    args = parser.parse_args()
    change_export_format(args)
    print("Finish!")