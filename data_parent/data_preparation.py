## Duplicates management


import os
import hashlib

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
