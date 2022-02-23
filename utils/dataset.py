import os
import sys
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Adding the parent directory to the sys.path
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)
import config


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_label_csv():
    label = pd.read_csv(config.LABEL_CSV_PATH)['sentence'].tolist()
    return label


def load_npz(filepath):
    data = np.load(f'{filepath}.npz')["data"] / 255
    # data.shape = (video_length, crop_height, crop_width, channels)
    # Convert rgb to grayscale
    if data.shape[3] == 3:
        rgb_weights = [0.2989, 0.5870, 0.1140]
        visual_data = np.dot(data[:][..., :3], rgb_weights)
        # Repeat the last row to make all the data length equal
        visual_data = repeat_last_row(visual_data, 120)
        return visual_data
    else:
        visual_data = repeat_last_row(data, 120)
        return visual_data


def repeat_last_row(data, target_length):
    if data.shape[0] < target_length:
        rep = target_length - data.shape[0]
        last = np.repeat([data[-1]], repeats=rep, axis=0)
        expanded = np.vstack([data, last])
        return expanded
    elif data.shape[0] >= target_length:
        return data[:target_length]
    else:
        return data


def one_hot(array):
    unique, inverse = np.unique(array, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return onehot


def prepare_dataset():
    # Load visual data
    foldernames = []

    for _, dirs, files in os.walk(config.SRC_FOLDER_PATH):
        if len(dirs) > 0:
            foldernames = sorted(dirs)

    for folder in tqdm(foldernames, desc='Folder', bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'):
        # Check if dst folder exists
        # create_path(f'{config.SRC_FOLDER_PATH}/{folder}')

        filenames = []
        for _, dirs, files in os.walk(f'{config.SRC_FOLDER_PATH}/{folder}'):
            filenames = sorted(file.split(".")[0] for file in list(
                filter(lambda x: x != ".DS_Store", files)))

            X = []
            y = np.identity(20)
            # y = load_label_csv()
            # print(y)
            # print(one_hot(y))

            for filename in tqdm(filenames[:20], desc='Files ', bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'):
                X.append(
                    load_npz(f'{config.SRC_FOLDER_PATH}/{folder}/{filename}').reshape((120, 96, 96, 1)))

            X_train, X_test, y_train, y_test = train_test_split(
                np.array(X), y, test_size=config.TEST_SPLIT_SIZE)

            # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
            append2h5py(X_train, X_test, y_train, y_test)


def append2h5py(X_train, X_test, y_train, y_test):
    create_path(config.INPUT_DATA_PATH)

    with h5py.File(config.H5FILE, "a") as h5file:
        # Save X_train data
        if not "X_train" in h5file:
            h5file.create_dataset("X_train", data=X_train,
                                  compression="gzip", chunks=True, maxshape=(None, 120, 96, 96, 1))
        else:
            h5file["X_train"].resize(
                (h5file["X_train"].shape[0] + X_train.shape[0]), axis=0)
            h5file["X_train"][-X_train.shape[0]:] = X_train

        # Save X_test data
        if not "X_test" in h5file:
            h5file.create_dataset("X_test", data=X_test,
                                  compression="gzip", chunks=True, maxshape=(None, 120, 96, 96, 1))
        else:
            h5file["X_test"].resize(
                (h5file["X_test"].shape[0] + X_test.shape[0]), axis=0)
            h5file["X_test"][-X_test.shape[0]:] = X_test

        # Save y_train data
        if not "y_train" in h5file:
            h5file.create_dataset("y_train", data=y_train,
                                  compression="gzip", chunks=True, maxshape=(None, 20))
        else:
            h5file["y_train"].resize(
                (h5file["y_train"].shape[0] + y_train.shape[0]), axis=0)
            h5file["y_train"][-y_train.shape[0]:] = y_train

        # Save y_test data
        if not "y_test" in h5file:
            h5file.create_dataset("y_test", data=y_test,
                                  compression="gzip", chunks=True, maxshape=(None, 20))
        else:
            h5file["y_test"].resize(
                (h5file["y_test"].shape[0] + y_test.shape[0]), axis=0)
            h5file["y_test"][-y_test.shape[0]:] = y_test

        h5file.close()


class DataLoader():

    @staticmethod
    def load_data():
        '''
        Load the dataset
            This dataset consists of X_train,
            y_train, X_test and y_test, four `.npy` files. We will save
            the dataset into `.npy` file.
        '''

        hdf5_file = config.H5FILE

        X_train = tfio.IODataset.from_hdf5(
            hdf5_file, dataset="/X_train")
        y_train = tfio.IODataset.from_hdf5(
            hdf5_file, dataset="/y_train")
        X_test = tfio.IODataset.from_hdf5(
            hdf5_file, dataset="/X_test")
        y_test = tfio.IODataset.from_hdf5(
            hdf5_file, dataset="/y_test")

        train = tf.data.Dataset.zip((X_train, y_train)).batch(
            config.BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        test = tf.data.Dataset.zip((X_test, y_test)).batch(
            config.BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

        return train, test


if __name__ == '__main__':
    # print(load_label_csv())
    # load_npz()
    prepare_dataset()
    # train, test = DataLoader().load_data()
    # print(train, test)
