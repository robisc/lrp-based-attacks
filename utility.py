import tensorflow as tf
import numpy as np

def load_dataset(data:str):
    """Load dataset from keras datasets

    Args:
        data (str): either "MNIST" or "CIFAR10

    Raises:
        ValueError: wrong dataset chosen

    Returns:
        x_train, y_train, x_test, y_test: tuple of training & test data and labels
    """
    # check if valid dataset was chosen
    if data not in ["MNIST", "CIFAR10"]:
        raise ValueError("Invalid dataset chosen")

    # respective data from keras
    if data == "MNIST":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif data == "CIFAR10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Now we extend / preprocess our data
    x_train = (((x_train/255)*2)-1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = (((x_test/255)*2)-1)
    x_test = np.expand_dims(x_test, axis=-1)
    y_train = np.eye(np.max(y_train)+1)[y_train]
    y_test = np.eye(np.max(y_test)+1)[y_test]

    return (x_train, y_train, x_test, y_test)

# Loading or training a model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.models import Model

def create_cnn(data: str):
    """Creates the keras CNN for either dataset

    Args:
        data (str): either MNIST or CIFAR10

    Raises:
        ValueError: Raises if wrong dataset was entered

    Returns:
        keras.model: CNN that is used for further steps
    """
    # check if valid dataset was chosen
    if data not in ["MNIST", "CIFAR10"]:
        raise ValueError("Invalid dataset chosen")

    if data == "MNIST":
        inputs =  Input(shape=(28,28,1))
        conv1 = Conv2D(32,(3,3),activation="relu", padding="same")(inputs)
        conv2 = Conv2D(32,(3,3),activation="relu", padding="same")(conv1)
    #     bn1 = BatchNormalization()(conv2) # remove
        pool1 = MaxPooling2D(pool_size=(2,2))(conv2)
        
        conv3 = Conv2D(64,(3,3),activation="relu", padding="same")(pool1)
        conv4 = Conv2D(64,(3,3),activation="relu", padding="same")(conv3)
    #     bn2 = BatchNormalization()(conv4) # remove
        pool2 = MaxPooling2D(pool_size=(2,2))(conv4)    

        flat = Flatten()(pool2)
        dense1 = Dense(units=128, activation ="sigmoid")(flat)
        output = Dense(units=10, activation ="softmax")(dense1)
        classifier = Model(inputs = inputs, outputs = output)
    elif data == "CIFAR10":
        inputs =  Input(shape=(32,32,3))
        conv1 = Conv2D(32,(3,3),activation="relu", padding="same")(inputs)
        conv2 = Conv2D(32,(3,3),activation="relu", padding="same")(conv1)
        pool1 = MaxPooling2D(pool_size=(2,2))(conv2)
        
        conv3 = Conv2D(64,(3,3),activation="relu", padding="same")(pool1)
        conv4 = Conv2D(64,(3,3),activation="relu", padding="same")(conv3)
        pool2 = MaxPooling2D(pool_size=(2,2))(conv4)    
        
        conv5 = Conv2D(128,(3,3),activation="relu", padding="same")(pool2)
        conv6 = Conv2D(128,(3,3),activation="relu", padding="same")(conv5)
        pool3 = MaxPooling2D(pool_size=(2,2))(conv6)
        
        flat = Flatten()(pool3)
        dense1 = Dense(units=128, activation ="sigmoid")(flat)
    #     drop1 = Dropout(rate=0.2)(dense1)
        output = Dense(units=10, activation ="softmax")(dense1)
        classifier = Model(inputs = inputs, outputs = output)
    
    # Compile classifier and adding optimizer
    classifier.compile(optimizer=tf.keras.optimizers.Adam(1e-03), 
                        loss = ["binary_crossentropy"],
                        metrics=["accuracy",tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
    
    return classifier