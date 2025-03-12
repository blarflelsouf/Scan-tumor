import numpy as np
import pandas as pd
from tensorflow.keras import Sequential, layers
from tensorflow.keras import models
from tensorflow.keras import metrics



# Import des images preprocess






# Shape des images
in_shape = (224, 224, 3)

# Debut du model
model = Sequential([
    layers.Conv2D(16, (3,3), padding='same', activation="relu", input_shape=in_shape),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Conv2D(32, (2,2), padding='same', activation="relu"),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dense(50, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

# Compilation du model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[metrics.Recall()])
