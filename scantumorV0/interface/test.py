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

print(data_train)

                  ######### Preprocess data #########

'''Preprocess the data'''

### Variable for preprocessing data ###

batch_size_prepro=32 # -> batch size of preprocessing
img_size = (150, 150) # -> choose the size of picture


### Prepropress the data by resizing, padding ###

#data_train_prepro = prepro.preprocess(data_train, batch_size_prepro, img_size) # Stock the data preprocess in a variable data_train_prepro
#data_test_prepro = prepro.preprocess(data_test, batch_size_prepro, img_size) # Stock the data preprocess in a variable data_test_prepro



                    ######### Data augmentation #########

'''Complete the dataset train with new picture for data augmentation, variables are:
    -
'''

### Variable for data augmentation ###

n = 3000 # -> nbr of picture generated of non-tumor
augdir=r'/home/blqrf/code/blarflelsouf/Scan-tumor/notebooks/raw_data' # directory to store the images if it does not exist it will be created

### Data augmentation ###

data_augment.make_and_store_images(data_train, augdir, n, img_size, color_mode='rgb', save_prefix='aug-',save_format='jpg')
