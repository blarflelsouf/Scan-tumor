import ml_logic.data as data
import ml_logic.modelCNN as modelCNN
import ml_logic.preprocessor as prepro
import ml_logic.modelvgg16 as modelvgg
import matplotlib as plt




###### V0 ######

'''
Load the data in the model from local//online and with dataframe//dataset

'''
### Variable for loading data ###
local = True # -> True for local // False for Download
path_train = '/home/blqrf/code/blarflelsouf/Scan-tumor/notebooks/Training' # -> Path to train file
path_test = '/home/blqrf/code/blarflelsouf/Scan-tumor/notebooks/Testing' # -> Path to train file
type_data = True # -> True for Dataframe // False for Dataset

### Load data ###
data_import = data.load_data(local, path_train, path_test, type_data)
data_train = data_import[0] # Data Train
data_test = data_import[1] # Data Test


### Preprocess ###
batch_size=32
img_size = (224, 224)

data_train_prepro = prepro.preprocess(data_train, batch_size, img_size)
data_test_prepro = prepro.preprocess(data_test, batch_size, img_size)


'''
Train a model and report his history
'''
### Variable for loading data ###
batch_size = 32
patience = 2
epochs = 20



### Train model ###

#CNN#
#historyCNN = modelCNN.model_train(data_train_prepro, batch_size, patience, epochs)


#VGG16#
historyVGG = modelvgg.train_model(data_train_prepro, epochs, patience, batch_size)

print(historyVGG)



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
