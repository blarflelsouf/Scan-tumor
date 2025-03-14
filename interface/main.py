from ml_logic import preprocessor as prepro
from ml_logic import modelVGG as modelvgg
from ml_logic import modelBINARY as modelbin
from ml_logic import data_preparation as prepa
import utils


                        ###### Preparation of the data ######

''' Remove the duplicate in each files '''

path_train = 'data_parent/A_raw_data/Training' # -> Where the raw data train is
path_test = 'data_parent/A_raw_data/Testing'  # -> Where the raw data train is

# Remove from directory the duplicate picture by comparing them
prepa.data_preparation(path_train)

# Remove from directory the duplicate picture by comparing them
prepa.data_preparation(path_test)




                        ######### Preprocess data #########

'''Preprocess the data, rezising and padding'''

### Variable for preprocessing data ###

path_train_prepro = 'data_parent/B_preprocess_data/training_prepro' # -> Where saving the processed data test
path_test_prepro = 'data_parent/B_preprocess_data/testing_prepro' # -> Where Saving the processed data test
img_size = (150, 150) # -> Size of the picture normalized
df_train = utils.load_data_dataframe(path_train) # -> Put the images_paths and lables of train data in a dataframe
df_test = utils.load_data_dataframe(path_test) # -> Put the images_paths and lables of test data in a dataframe


### Prepropress the data by resizing, padding ###

# Stock the data preprocess in a directory data_train_prepro
prepro.preprocess_write_squared_image_to_dir(df_train, img_size, root_dest_dir= path_train_prepro)

# Stock the data preprocess in a directory data_test_prepro
prepro.preprocess_write_squared_image_to_dir(df_test, img_size, root_dest_dir= path_test_prepro)






                        ######### Binary model #########
''' Use a VGG16 model to categorized the data, metric used is recall'''
### Variable for loading data ###

batch_size_train = 256 # -> batch size of training model (128/256 recommended)
patience = 2 # -> patience of early stopping
epochs = 20 # -> number of epochs
nbr_img = 2500 # -> Nbr of pic no_tumor pic added to the data

### Train model ###

# history_bin = modelbin.train_model_bin(path_train_prepro, path_test_prepro, nbr_img, img_size, patience, epochs, batch_size_train) #Binary#

# histo_bin_train = history_bin[0]
# recall_bin_train = histo_bin_train.__dict__['history']['recall']

# histo_bin_test = history_bin[1]
# print('⭐ Binary model: Recall on train dataset: ', recall_bin_train.mean())
# print('⭐ Binary model: Recall on test dataset: ', histo_bin_test[1])

# # Model bin fitted
# model_bin = history_bin[2]






                        ######### Categorical model #########

''' Use a VGG16 model to categorized the data, metric used is accuracy'''
### Variable for loading data ###

batch_size_train = 256 # -> batch size of training model (64/128 recommended)
patience = 2 # -> patience of early stopping
epochs = 20 # -> number of epochs

### Train model ###

history_cat = modelvgg.train_model_cat(path_train_prepro, path_test_prepro, epochs, patience, batch_size_train, img_size) #VGG16#

histo_cat_train = history_cat[0]
histo_cat_train = histo_cat_train.__dict__['accuracy'].mean()

histo_cat_test = history_cat[1]
print('⭐ Accuracy on train dataset: ', histo_cat_train)
print('⭐ Accuracy on test dataset: ', histo_cat_test)

# Model fitted
model_cat= history_cat[2]
