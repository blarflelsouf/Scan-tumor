import os
import pandas as pd


# Function to determine label from path
def get_label(path):
    if 'notumor' in path:
        return 'notumor'
    elif 'glioma' in path:
        return 'glioma'
    elif 'meningioma' in path:
        return 'meningioma'
    elif 'pituitary' in path:
        return 'pituitary'
    else:
        return 'No label found'

# Function to create a data frame
def load_data_to_df(train_data_dir):
    img_list = os.listdir(train_data_dir)
    images_paths_train =[]

    for image in img_list :
        image_path = os.path.join(train_data_dir, image)
        images_paths_train.append(image_path)

    df = pd.DataFrame(images_paths_train, columns=['images_paths'])
    df['labels'] = df['images_paths'].apply(get_label)
    return df
