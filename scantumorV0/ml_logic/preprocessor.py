from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def preprocess(data, batch_size, img_size):
    print('preprocess in progress')

    gen = ImageDataGenerator()
    train_gen = gen.flow_from_dataframe(
        data,
        x_col= 'images_paths',
        y_col= 'labels', target_size=img_size,
        class_mode= 'categorical',
        color_mode= 'rgb',
        shuffle = True,
        batch_size= batch_size
    )


    print('preprocessed completed')
    return train_gen
