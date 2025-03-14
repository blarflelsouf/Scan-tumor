## Duplicates management

import os
import hashlib
import shutil
import ml_logic.data_augment as data_augment
import utils



'''
def merge_files(directory_path = 'data_parent/A_raw_data',dest_dir = 'data_parent/B_raw_data_merged_dedup'):
    list_dir=[f.path for f in os.scandir(directory_path) if f.is_dir()]
    for folder in list_dir:
        for file in os.listdir(folder):
            src_file = os.path.join(folder, file)
            dest_path = os.path.join(dest_dir, file)
            shutil.move(src_file, dest_path)
'''



def compute_hash(file):
    hasher = hashlib.md5()
    with open(file, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


# This function creates a dictionary with hash and paths of the files

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


# This function to display the duplicates

def display_duplicates(directory_path):
    hash_dict = listfiles(directory_path)
    duplicates = {hash_value: paths for hash_value, paths in hash_dict.items() if len(paths) > 1}
    total_duplicates = sum(len(paths) - 1 for paths in duplicates.values())
    print(f'There are {total_duplicates} duplicate files to remove in {directory_path}')
    for hash_value, paths in duplicates.items():
        print(f"Duplicate files with hash {hash_value}:")
        for path in paths:
            print(f" - {path}")



# This function to list the duplicates

def find_duplicates(directory_path):
    hash_dict = listfiles(directory_path)
    duplicates = {hash_value: paths for hash_value, paths in hash_dict.items() if len(paths) > 1}
    return duplicates



# This function to remove the duplicates listed from the function above

def remove_duplicates(directory_path):
    duplicates = find_duplicates(directory_path)
    for hash_value, paths in duplicates.items():
        print(f"Removing duplicates for hash {hash_value}:")
        for path in paths[1:]:
            print(f" - Removing {path}")
            os.remove(path)



def data_preparation(path: str):

    # Remove the duplicate in each children folder
    print('🧮 Removing duplicate in raw data in process')

    remove_duplicates(path + '/glioma')
    remove_duplicates(path + '/meningioma')
    remove_duplicates(path + '/no_tumor')
    remove_duplicates(path + '/pituitary')


    print('⭐ Deleting duplicate is completed!')

    return None
