# importing libraries
import tensorflow as tf
from tensorflow import keras

def create_cnn_architecture_model1(input_shape):
    '''
    The structure of the model is defined and it is compiled and returned.
    INPUT : shape of the image to be given to the model.
    OUTPUT: model after compilationn
    '''
    inp = keras.layers.Input(shape=input_shape)

    conv1 = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),
                                activation='relu', padding='same')(inp)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                                   activation='relu', padding='same')(pool1)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    flat = keras.layers.Flatten()(pool2)

    hidden1 = keras.layers.Dense(256, activation='relu')(flat)
    drop1 = keras.layers.Dropout(rate=0.3)(hidden1)

    out = keras.layers.Dense(10, activation='softmax')(drop1)

    model = keras.Model(inputs=inp, outputs=out)
    # using adam optimizer, sparse_categorical_crossentropy because its a multiclass classification
    # the metrices used here is accuracy.
    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    return model
def create_(INPUT_SHAPE):
    ''' for calling the model architecture.
    INPUT : Shape of the image
    OUTPUT : Model architecture. 
    '''

    model = create_cnn_architecture_model1(input_shape=INPUT_SHAPE)
    return model
