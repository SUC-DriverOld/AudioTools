import os
import shutil
import argparse

def copy_svp_files(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(('.svp', '.s5p')):
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_dir, file)

                shutil.copy(source_path, target_path)
                print(f"Copy {source_path} to {target_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='input', help='Input directory')
    parser.add_argument('-o', '--output_dir', type=str, default='output', help='Output directory')
    args = parser.parse_args()
    copy_svp_files(args.input_dir, args.output_dir)
    print("Finish!")