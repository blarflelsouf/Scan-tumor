
import os
import hashlib
import shutil
import data_augment
import scantumorV0.ml_logic.duplicates_manag as duplicates_manag
import utils
import load_data

# 1st step is to remove duplicates direcly on A_raw_data

duplicates_manag.remove_duplicates('data_parent/A_raw_data/Training/glioma')
duplicates_manag.remove_duplicates('data_parent/A_raw_data/Training/meningioma')
duplicates_manag.remove_duplicates('data_parent/A_raw_data/Training/no_tumor')
duplicates_manag.remove_duplicates('data_parent/A_raw_data/Training/pituitary')
duplicates_manag.remove_duplicates('data_parent/A_raw_data/Testing/glioma')
duplicates_manag.remove_duplicates('data_parent/A_raw_data/Testing/meningioma')
duplicates_manag.remove_duplicates('data_parent/A_raw_data/Testing/no_tumor')
duplicates_manag.remove_duplicates('data_parent/A_raw_data/Testing/pituitary')

# 2nd step is to augment data (at a class level -> notumor here)
data_to_augment = 'data_parent/A_raw_data/Training/notumor'

data_augment.make_and_store_images(
    data_train = load_data.load_data_to_df(data_to_augment),
    augdir = 'data_parent/B_data_augmented',
    n = 1696,
    color_mode='rgb',
    save_format='png')

# 3rd step is to manually move to C_data_to_process:
    # - all augmented data (from B_data_augmented)
    # - all data from Training (from A_raw_data)
