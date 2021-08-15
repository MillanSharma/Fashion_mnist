# importing libraries
import tensorflow as tf
from tensorflow import keras
import model_dispatcher
import argparse
import os
import inference
INPUT_SHAPE = (28, 28, 1) #shape of input shape
def run():
    ''' fitting, training, fitting and savinig weights
    Input :
    Output:
    '''
    # loading the dataset from keras dataset module
    fashion_mnist = keras.datasets.fashion_mnist
    # train test split
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    # classes present in the dataset.
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # reshaping to fit in the training model.
    train_images_gr = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images_gr = test_images.reshape(test_images.shape[0], 28, 28, 1)
    EPOCHS = 5
    train_images_scaled = train_images_gr / 255.
    f_model = model_dispatcher.create_(INPUT_SHAPE)
    # fitting on the dataset, 
    f_model.fit(train_images_scaled, train_labels, validation_split=0.2, epochs=EPOCHS)
    if not os.path.isdir('model/'):
        os.mkdir('model/')
    f_model.save_weights(filepath='model/cnn_model1_wt.h5', overwrite=True)

if __name__ == "__main__":
    run()
