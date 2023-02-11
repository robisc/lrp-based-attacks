import tensorflow as tf
import numpy as np
import os
import pandas as pd

def load_dataset(dataset:str):
    """Load dataset from keras datasets

    Args:
        data (str): either "MNIST" or "CIFAR10

    Raises:
        ValueError: wrong dataset chosen

    Returns:
        x_train, y_train, x_test, y_test: tuple of training & test data and labels
    """
    # check if valid dataset was chosen
    if dataset not in ["MNIST", "CIFAR10"]:
        raise ValueError("Invalid dataset chosen")

    # respective data from keras
    if dataset == "MNIST":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset == "CIFAR10":
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

def create_cnn(dataset: str):
    """Creates the keras CNN for either dataset

    Args:
        data (str): either MNIST or CIFAR10

    Raises:
        ValueError: Raises if wrong dataset was entered

    Returns:
        keras.model: CNN that is used for further steps
    """
    # check if valid dataset was chosen
    if dataset not in ["MNIST", "CIFAR10"]:
        raise ValueError("Invalid dataset chosen")

    if dataset == "MNIST":
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
    elif dataset == "CIFAR10":
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

def get_classifier(dataset: str, mode: str, data = "", save_model: bool = False):
    # Do input validation
    if dataset not in ["MNIST", "CIFAR10"]:
        raise ValueError("Invalid dataset chosen")
    if mode not in ["load", "train"]:
        raise ValueError("Invalid mode chosen")
    
    if mode == "train":
        # First we check if the data was entered as tuple
        if type(data) != tuple:
            raise TypeError("Input datas as tuple")

        # Then we create the respective keras model
        print("Creating classifier ...")
        classifier = create_cnn(dataset = dataset)

        # Now we use the training data
        (x_train, y_train, x_test, y_test) = data

        # We use SSS to split our set into random groups
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
        (train, val) = list(sss.split(x_train,y_train))[0]

        # Preprocessing and datagenerator
        print("Training datagenerator ...")
        from keras.preprocessing.image import ImageDataGenerator
        datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                zoom_range = 0, # Randomly zoom image 
                width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=False,  # randomly flip images
                vertical_flip=False)  # randomly flip images
        datagen.fit(x_train)

        # Path to model
        model_path = os.path.join("models", dataset, "model1.ckpt")
        model_dir = os.path.dirname(model_path)

        # Some configs for the datasets
        batch_size = {"MNIST": 2000, "CIFAR10": 500}[dataset]
        epochs = {"MNIST": 10, "CIFAR10": 30}[dataset]

        # Learning loop with collecting history
        print("Starting training loop ...")
        hist =  classifier.fit(datagen.flow(x_train[train], y_train[train], batch_size=batch_size), epochs = epochs,
                                        validation_data = (x_train[val,:], y_train[val,:]))
        
        keys = hist.history.keys()

        history = pd.DataFrame(hist.history["val_loss"])
        history.columns = ["val_loss"]
        history["loss"] = hist.history["loss"]
        history["recall"] = hist.history["recall"]
        history["val_recall"] = hist.history["val_recall"]
        history["precision"] = hist.history["precision"]
        history["val_precision"] = hist.history["val_precision"]

        history.plot(title = f"Training results on dataset {dataset}")

        if save_model:
            print("Saving model ...")
            tf.keras.models.save_model(classifier, f"{dataset}.hdf5")
        
        