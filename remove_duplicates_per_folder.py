import hashlib
import os

# Function to compute the hash of a file
def compute_hash(file):
    hasher = hashlib.md5()
    with open(file, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Function to list the files with their hash
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

# Function to display the duplicates (hash and paths)
def display_duplicates(directory_path):
    hash_dict = listfiles(directory_path)
    duplicates = {hash_value: paths for hash_value, paths in hash_dict.items() if len(paths) > 1}
    total_duplicates = sum(len(paths) - 1 for paths in duplicates.values())
    print(f'There are {total_duplicates} duplicate files to remove in {directory_path}')
    for hash_value, paths in duplicates.items():
        print(f"Duplicate files with hash {hash_value}:")
        for path in paths:
            print(f" - {path}")

duplicates = display_duplicates('data/Training/pituitary')
duplicates

# Function to get and save the duplicates (hash and paths)
def find_duplicates(directory_path):
    hash_dict = listfiles(directory_path)
    duplicates = {hash_value: paths for hash_value, paths in hash_dict.items() if len(paths) > 1}
    return duplicates

# Function to remove the duplicates
def remove_duplicates(directory_path):
    duplicates = find_duplicates(directory_path)
    for paths in duplicates.items():
        for path in paths[1:]:
            os.remove(path)
