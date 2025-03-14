from tensorflow.keras.applications.vgg16 import VGG16
def load_model():
    model = VGG16(weights="imagenet", include_top=False, input_shape=(150,150,3))
    return model
def set_nontrainable_layers(model):
    model.trainable = False
    return model

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers, optimizers, callbacks, models
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping

def add_last_layers(model):
    '''Take a pre-trained model, set its parameters as non-trainable, and add additional trainable layers on top'''
    non_trainable_model = set_nontrainable_layers(model)


    # Convolutional Layers

    new_model = Sequential()
    new_model.add(non_trainable_model)
    new_model.add(layers.Input((150, 150, 3)))
    new_model.add(layers.Rescaling(1./255))
    new_model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same"))
    new_model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="same") )

    new_model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same"))
    new_model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="same") )

    new_model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"))
    new_model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="same") )

    new_model.add(layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu", padding="same"))
    new_model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="same") )

    new_model.add(layers.Flatten())

    new_model.add(layers.Dense(64, activation="relu"))

    new_model.add(layers.Dropout(0.5))
    new_model.add(layers.Dense(4, activation="softmax"))

    return new_model

def build_model():
    model = load_model()
    model = add_last_layers(model)

    opt = optimizers.Adam(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

def train_model_cat(path_train_prepro, path_test_prepro, epochs, patience, batch_size, img_size):
    print('🧮 Training of the model')
    train_ds = image_dataset_from_directory(
        path_train_prepro,
        labels="inferred",
        class_names=["notumor","glioma", 'meningioma', 'pituitary'],
        label_mode="categorical",
        seed=123,
        validation_split=0.25,
        subset='both',
        image_size=img_size,
        batch_size=batch_size)


    test_ds = image_dataset_from_directory(
        path_test_prepro,
        labels="inferred",
        class_names=["notumor","glioma", 'meningioma', 'pituitary'],
        label_mode="categorical",
        seed=123,
        image_size=img_size,
        batch_size=batch_size)


    model=build_model()

    es = EarlyStopping(
        patience=patience,
        restore_best_weights=True,
        verbose=2
    )


    history = model.fit(
            train_ds[0],
            epochs=epochs,
            callbacks=[es],
            batch_size=batch_size,
            validation_data=train_ds[1]
            )

    #print('🧮 Predict of the test')
    #pred = model.predict(test_ds)

    print('🧮 Evalutation of the model on test')
    scores = model.evaluate(test_ds)

    print('⭐ Return of the results')

    return history, scores
