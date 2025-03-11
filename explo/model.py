import numpy as np
import pandas as pd
from tensorflow.keras import Sequential, layers
from tensorflow.keras import models



# Import des images preprocess





# Debut du model

model = models.Sequential()

model.add(layers.Conv2D(6, kernel_size=(3, 3), activation='relu', input_shape=(225, 225, 3)))
model.add(layers.Conv2D(4, kernel_size=(3), activation='relu')) # kernel_size = 3 <==> (3, 3)
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))
