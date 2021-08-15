# loading all the libraries 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import model_dispatcher
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

INPUT_SHAPE = (28, 28, 1)
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images_gr = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images_gr = test_images.reshape(test_images.shape[0], 28, 28, 1)
def infer(test_images_gr):
    '''
    Takes test images and returs the result in form of confussion matrix.
    INPUT : Test images for inference
    OUTPUT: Confussion matrix after the inference on test data.
    '''
    model = model_dispatcher.create_(INPUT_SHAPE)
    # loading the weights
    model.load_weights('model/cnn_model1_wt.h5')
    print("model loaded")
    test_images_scaled = test_images_gr / 255.
    print("image scaled")
    # preditction from model
    predictions = model.predict(test_images_scaled)
    predictions[:5]
    prediction_labels = np.argmax(predictions, axis=1)
    prediction_labels[:5]
    # measuring the accuracy of the model on test set.
    acc = accuracy_score(test_labels, prediction_labels)
    print("accuracy is :",acc)
    pp=pd.DataFrame(confusion_matrix(test_labels, prediction_labels), index=class_names, columns=class_names)
    pp.to_csv("confussion_matrix.csv", index=False)

if __name__ == "__main__":
    infer(test_images_gr)
