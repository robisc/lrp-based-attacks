import pytest
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from keras.models import Model

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
    x_train = ((x_train/255)*2)-1
    x_test = ((x_test/255)*2)-1
    if dataset == "MNIST":
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

    y_train = np.eye(np.max(y_train)+1)[y_train]
    y_test = np.eye(np.max(y_test)+1)[y_test]
    if dataset == "CIFAR10":
        y_train = y_train.reshape((len(y_train),10))
        y_test = y_test.reshape((len(y_test),10))

    return (x_train, y_train, x_test, y_test)

# Loading or training a model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization

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
#        bn1 = BatchNormalization()(conv2) # remove
        pool1 = MaxPooling2D(pool_size=(2,2))(conv2)

        conv3 = Conv2D(64,(3,3),activation="relu", padding="same")(pool1)
        conv4 = Conv2D(64,(3,3),activation="relu", padding="same")(conv3)
#        bn2 = BatchNormalization()(conv4) # remove
        pool2 = MaxPooling2D(pool_size=(2,2))(conv4)

        flat = Flatten()(pool2)
        dense1 = Dense(units=128, activation ="sigmoid")(flat)
        output = Dense(units=10, activation ="softmax")(dense1)
        classifier = Model(inputs = inputs, outputs = output)
    elif dataset == "CIFAR10":
        inputs =  Input(shape=(32,32,3))
        conv1 = Conv2D(32,(3,3),activation="relu", padding="same")(inputs)
        conv2 = Conv2D(32,(3,3),activation="relu", padding="same")(conv1)
        bn1 = BatchNormalization()(conv2) # remove
        pool1 = MaxPooling2D(pool_size=(2,2))(bn1)

        conv3 = Conv2D(64,(3,3),activation="relu", padding="same")(pool1)
        conv4 = Conv2D(64,(3,3),activation="relu", padding="same")(conv3)
        bn2 = BatchNormalization()(conv4) # remove
        pool2 = MaxPooling2D(pool_size=(2,2))(bn2)

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
                        loss = ["categorical_crossentropy"],
                        metrics=["accuracy",tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

    return classifier

def get_classifier(dataset: str, mode: str, data = "", save_model: bool = False):
    """Returns a classifier CNN either from file or trains a new one

    Args:
        dataset (str): String of either MNIST or CIFAR10
        mode (str): either load or train
        data (str, optional): If train is chosen, contains all data as tuple 
                            (x_train, y_train, x_test, y_test). Defaults to "".
        save_model (bool, optional): Flag if model should be saved in current folder.

    Raises:
        ValueError: Invalid dataset
        ValueError: Invalid mode
        TypeError: invalid data input

    Returns:
        keras model: CNN for further use
    """
    # Do input validation
    if dataset not in ["MNIST", "CIFAR10"]:
        raise ValueError("Invalid dataset chosen")
    if mode not in ["load", "train"]:
        raise ValueError("Invalid mode chosen")

    if mode == "train":
        # First we check if the data was entered as tuple
        if type(data) != tuple:
            raise TypeError("You have to enter the data as tuple (x_train, y_train, x_test, y_test)")

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

        # Some configs for the datasets
        batch_size = {"MNIST": 2000, "CIFAR10": 500}[dataset]
        epochs = {"MNIST": 15, "CIFAR10": 30}[dataset]

        # Learning loop with collecting history
        print("Starting training loop ...")
        hist =  classifier.fit(datagen.flow(x_train[train], y_train[train],
                                        batch_size=batch_size),
                                        epochs = epochs,
                                        validation_data = (x_train[val,:], y_train[val,:]))

        print("Extracting history ...")
        keys = hist.history.keys()
        history = pd.DataFrame(hist.history[[i for i in keys if i.startswith("val_loss")][0]])
        history.columns = ["val_loss"]
        history["loss"] = hist.history[[i for i in keys if i.startswith("loss")][0]]
        history["recall"] = hist.history[[i for i in keys if i.startswith("recall")][0]]
        history["val_recall"] = hist.history[[i for i in keys if i.startswith("val_recall")][0]]
        history["precision"] = hist.history[[i for i in keys if i.startswith("precision")][0]]
        history["val_precision"] = hist.history[[i for i in keys if i.startswith("val_precision")][0]]
        history["accuracy"] = hist.history[[i for i in keys if i.startswith("accuracy")][0]]
        history["val_accuracy"] = hist.history[[i for i in keys if i.startswith("val_accuracy")][0]]
        history.plot(title = f"Training results on dataset {dataset}")
        y_pred = classifier.predict(x_test)
        print("The F1-Score on x_test is :" +str({f1_score(np.argmax(y_test, axis = 1),\
                                            np.argmax(get_onehot_argmax(y_pred,10),axis = 1),\
                                            average = 'macro')}))

        if save_model:
            print("Saving model ...")
            tf.keras.models.save_model(classifier, f"{dataset}.hdf5")
    elif mode == "load":
        print("Loading model ...")
        classifier = tf.keras.models.load_model(f"{dataset}.hdf5")

    return classifier


def get_onehot_argmax(target, num_classes: int):
    """Returns a matrix in onehot encoding with num_classes columns

    Args:
        target (): Variable containing your predictions
        num_classes (int): number of maximum classes (important if a class is not in the target)

    Returns:
        _type_: Onehot matrix
    """
    res = np.array(list(map(np.argmax, target)))
    res = np.eye(num_classes)[res]
    return res

def test_onehot_argmax_for_multiple_inputs():
    assert (~(get_onehot_argmax([[0.1,0,0,0],[0,0.9,0.1,0],[0,0.9,1,0],[0,0,0,1]], 4) == np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])) ).sum() == 0


def transform_image(img):
    """Transforms image from [-1, ... 1] to [0, ..., 1]

    Args:
        img (_type_): Input image

    Returns:
        _type_: Output Image
    """
    return (img+1)/2
