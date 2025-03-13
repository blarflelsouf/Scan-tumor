from ml_logic import preprocessor as prepro
from ml_logic import modelVGG as modelvgg
import utils



path_train = '/home/blqrf/code/blarflelsouf/Scan-tumor/notebooks/Training'
path_test = '/home/blqrf/code/blarflelsouf/Scan-tumor/notebooks/Testing'
path_train_prepro = '/home/blqrf/code/blarflelsouf/Scan-tumor/notebooks/training_prepro'
path_test_prepro = '/home/blqrf/code/blarflelsouf/Scan-tumor/notebooks/testing_prepro'







                        ######### Preprocess data #########

'''Preprocess the data'''

### Variable for preprocessing data ###

img_size = (150, 150)
df_train = utils.load_data_dataframe(path_train)
df_test = utils.load_data_dataframe(path_test)


### Prepropress the data by resizing, padding ###

# Stock the data preprocess in a directory data_train_prepro
prepro.preprocess_write_squared_image_to_dir(df_train, img_size, root_dest_dir= path_train_prepro)

# Stock the data preprocess in a directory data_test_prepro
prepro.preprocess_write_squared_image_to_dir(df_test, img_size, root_dest_dir= path_test_prepro)





### Variable for loading data ###

batch_size_train = 512 # -> batch size of training model (64/128 recommended)
patience = 1 # -> patience of early stopping
epochs = 5 # -> number of epochs
model = 'VGG' # -> 'CNN'//'VGG'//'ALL'


### Train model ###


if model == 'VGG':
    history = modelvgg.train_model(path_train_prepro, path_test_prepro, epochs, patience, batch_size_train, img_size) #VGG16#

histo_train = history[0]
histo_train = histo_train.__dict__['accuracy'].mean()

histo_test = history[1]
print('⭐ Accuracy on train dataset: ', histo_train)
print('⭐ Accuracy on test dataset: ', histo_test)
