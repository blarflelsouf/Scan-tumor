import ml_logic.data as data
import ml_logic.modelCNN as modelCNN
import ml_logic.preprocessor as prepro
import ml_logic.modelvgg16 as modelvgg
import ml_logic.data_augment as data_augment
import pandas as pd



################################## V0 ##################################



                    ######## Loading data ########

'''Load the data in the model from local//online and with dataframe//dataset'''


### Variable for loading data ###

local = True # -> True for local // False for Download
path_train = '/home/blqrf/code/blarflelsouf/Scan-tumor/notebooks/Training' # -> Path to train file
path_test = '/home/blqrf/code/blarflelsouf/Scan-tumor/notebooks/Testing' # -> Path to test file
type_data = True # -> True for Dataframe // False for Dataset
nb_image_test = 10 # -> Nbr of image to test


### Load data  ###

data_import = data.load_data(local, path_train, path_test, type_data) # Call the load-data file
data_train = data_import[0] # Stock the data in a variable data_train
data_train2 = data_import[1]
data_train = pd.concat([data_train, data_train2])

data_test = data_train.sample(nb_image_test)  # Stock a sample of the data in a variable data_test
a= list(data_test.index)
data_train.drop(a, axis=0, inplace=True) # Drop datatest in datatrain



                    ######### Data augmentation #########

'''Complete the dataset train with new picture for data augmentation, variables are:
    -
'''

### Variable for data augmentation ###

n = 3000 # -> nbr of picture generated of non-tumor
augdir=r'/home/blqrf/code/blarflelsouf/Scan-tumor/notebooks/raw_data' # directory to store the images if it does not exist it will be created
img_size = (224, 224) # -> choose the size of picture

### Data augmentation ###

data_augment.make_and_store_images(data_train, augdir, n, img_size, color_mode='rgb', save_prefix='aug-',save_format='jpg')



                  ######### Preprocess data #########

'''Preprocess the data'''

### Variable for preprocessing data ###

batch_size_prepro=32 # -> batch size of preprocessing
img_size = (224, 224) # -> choose the size of picture


### Prepropress the data by resizing, padding ###

data_train_prepro = prepro.preprocess(data_train, batch_size_prepro, img_size) # Stock the data preprocess in a variable data_train_prepro
data_test_prepro = prepro.preprocess(data_test, batch_size_prepro, img_size) # Stock the data preprocess in a variable data_test_prepro




                ######### Train a model #########

'''Train a model and report his history. You can choose between:
    - the batch size of training
    - the patience of early stopping
    - the number of epochs
    - the type of model: CNN // VGG
'''

### Variable for loading data ###

batch_size_train = 128 # -> batch size of training model (64/128 recommended)
patience = 1 # -> patience of early stopping
epochs = 5 # -> number of epochs
model = '' # -> 'CNN'//'VGG'//'ALL'
history_all = []


### Train model ###

if model == 'CNN':
    history = modelCNN.model_train(data_train_prepro, batch_size_train, patience, epochs) #CNN#

elif model == 'VGG':
    history = modelvgg.train_model(data_train_prepro, epochs, patience, batch_size_train) #VGG16#

elif model == 'ALL':
    history_all.append(modelvgg.train_model(data_train_prepro, epochs, patience, batch_size_train)) #VGG16#
    history_all.append(modelCNN.model_train(data_train_prepro, batch_size_train, patience, epochs)) #CNN#



                ######### Evaluate a model #########




'''
def plot_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('accuracy')
    ax[1].plot(history.epoch, history.history["accuracy"], label="Train acc")
    ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()

plot_history(history)
'''
