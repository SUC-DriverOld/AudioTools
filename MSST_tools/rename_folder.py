import os

def rename_subfolders(parent_folder):
    subfolders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]

    subfolders.sort()

    for index, subfolder in enumerate(subfolders, start=1):
        old_name = os.path.join(parent_folder, subfolder)
        new_name = os.path.join(parent_folder, str(index))

        os.rename(old_name, new_name)
        print(f"Renamed: {old_name} -> {new_name}")

parent_folder = 'datasets/valid'
rename_subfolders(parent_folder)
