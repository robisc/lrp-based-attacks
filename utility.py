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