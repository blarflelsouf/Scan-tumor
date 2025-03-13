
import os
import hashlib
import shutil


## Merge management --> from A_raw_data to B_raw_data_merged


def merge_files(directory_path = 'data_parent/A_raw_data',dest_dir = 'data_parent/B_raw_data_merged_dedup'):
    list_dir=[f.path for f in os.scandir(directory_path) if f.is_dir()]
    for folder in list_dir:
        for file in os.listdir(folder):
            src_file = os.path.join(folder, file)
            dest_path = os.path.join(dest_dir, file)
            shutil.move(src_file, dest_path)


## Duplicates management --> from/ to B_raw_data_merged_dedup


def compute_hash(file):
    hasher = hashlib.md5()
    with open(file, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def listfiles(directory_path):
    hash_dict = {}
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_hash = compute_hash(file_path)
            if file_hash in hash_dict:
                hash_dict[file_hash].append(file_path)
            else:
                hash_dict[file_hash] = [file_path]
    return hash_dict

def display_duplicates(directory_path):
    hash_dict = listfiles(directory_path)
    duplicates = {hash_value: paths for hash_value, paths in hash_dict.items() if len(paths) > 1}
    total_duplicates = sum(len(paths) - 1 for paths in duplicates.values())
    print(f'There are {total_duplicates} duplicate files to remove in {directory_path}')
    for hash_value, paths in duplicates.items():
        print(f"Duplicate files with hash {hash_value}:")
        for path in paths:
            print(f" - {path}")

def find_duplicates(directory_path):
    hash_dict = listfiles(directory_path)
    duplicates = {hash_value: paths for hash_value, paths in hash_dict.items() if len(paths) > 1}
    return duplicates

def remove_duplicates(directory_path):
    duplicates = find_duplicates(directory_path)
    for hash_value, paths in duplicates.items():
        print(f"Removing duplicates for hash {hash_value}:")
        for path in paths[1:]:
            print(f" - Removing {path}")
            os.remove(path)


## Data split --> from B_raw_data_merged_dedup to C_raw_data_split
# Once data is merged and duplicates are removed, we can split the data into training and testing sets according to their name

def split_data_name(source_directory_path):
    for file_name in source_directory_path:
        src_file = os.path.join(source_directory_path, file_name)
        dest_path = os.path.join(dest_directory_path, file_name)
        if 'TR-gl' in file_name: # training glioma
            dest_directory_path = 'data_parent/C_data_split/Training/glioma'
            shutil.move(src_file, dest_path)
        elif 'TE-gl' in file_name:
            dest_directory_path = 'data_parent/C_data_split/Testing/glioma'
            shutil.move(src_file, dest_path)
        elif 'TR-pi' in file_name:
            dest_directory_path = 'data_parent/C_data_split/Training/pituitary'
            shutil.move(src_file, dest_path)
        elif 'TE-pi' in file_name:
            dest_directory_path = 'data_parent/C_data_split/Testing/pituitary'
            shutil.move(src_file, dest_path)
        elif 'TR-me' in file_name:
            dest_directory_path = 'data_parent/C_data_split/Training/meningioma'
            shutil.move(src_file, dest_path)
        elif 'TE-me' in file_name:
            dest_directory_path = 'data_parent/C_data_split/Testing/meningioma'
            shutil.move(src_file, dest_path)
        elif 'TR-no' in file_name:
            dest_directory_path = 'data_parent/C_data_split/Training/no_tumor'
            shutil.move(src_file, dest_path)
        elif 'TE-no' in file_name:
            dest_directory_path = 'data_parent/C_data_split/Testing/no_tumor'
            shutil.move(src_file, dest_path)
        else:
            print('Error: file name not recognized')
            break


## Data augmented --> from C_raw_data_split to D_raw_data_augmented



## Data preprocessed --> from D_raw_data_augmented to E_raw_data_preprocessed
