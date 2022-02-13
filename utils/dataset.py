import os
import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    label = pd.read_csv(config.LABEL_CSV_PATH)
    label = np.array(label)
    return label


def load_npz(filepath):
    data = np.load(f'{filepath}.npz')["data"] / 255
    # data.shape = (video_length, crop_height, crop_width, channels)
    # Convert rgb to grayscale
    if data.shape[3] == 3:
        rgb_weights = [0.2989, 0.5870, 0.1140]
        visual_data = np.dot(data[:][..., :3], rgb_weights)
        # Repeat the last row to make all the data length equal
        visual_data = repeat_last_row(visual_data, 150)
        return visual_data
    else:
        visual_data = repeat_last_row(data, 150)
        return visual_data


def repeat_last_row(data, target_length):
    if data.shape[0] < target_length:
        rep = target_length - data.shape[0]
        last = np.repeat([data[-1]], repeats=rep, axis=0)
        expanded = np.vstack([data, last])
        return expanded
    else:
        return data


def prepare_dataset():
    # Load visual data
    foldernames = []

    for _, dirs, files in os.walk(config.SRC_FOLDER_PATH):
        if len(dirs) > 0:
            foldernames = sorted(dirs)

    for folder in tqdm(foldernames, desc='Folder', bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'):
        # Check if dst folder exists
        create_path(f'{config.SRC_FOLDER_PATH}/{folder}')

        filenames = []
        for _, dirs, files in os.walk(f'{config.SRC_FOLDER_PATH}/{folder}'):
            filenames = sorted(file.split(".")[0] for file in list(
                filter(lambda x: x != ".DS_Store", files)))

            X = []
            y = load_label_csv()

            for filename in tqdm(filenames, desc='Files ', bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'):
                X.append(
                    load_npz(f'{config.SRC_FOLDER_PATH}/{folder}/{filename}'))

            X_train, X_test, y_train, y_test = train_test_split(
                np.array(X), y, test_size=config.TEST_SPLIT_SIZE)

            append2h5py(X_train, X_test, y_train, y_test)


def append2h5py(X_train, X_test, y_train, y_test):
    h5file_path = f'{config.INPUT_DATA_PATH}/dataset.h5'

    with h5py.File(h5file_path, "a") as h5file:
        if not "X_train" in h5file:
            h5file.create_dataset("X_train", data=X_train,
                                  compression="gzip", chunks=True, maxshape=(None, 150, 96, 96))
        else:
            h5file["X_train"].resize(
                (h5file["X_train"].shape[0] + X_train.shape[0]), axis=0)
            h5file["X_train"][-X_train.shape[0]:] = X_train

        if not "X_test" in h5file:
            h5file.create_dataset("X_test", data=X_test,
                                  compression="gzip", chunks=True, maxshape=(None, 150, 96, 96))
        else:
            h5file["X_test"].resize(
                (h5file["X_test"].shape[0] + X_test.shape[0]), axis=0)
            h5file["X_test"][-X_test.shape[0]:] = X_test

        if not "y_train" in h5file:
            h5file.create_dataset("y_train", data=y_train,
                                  compression="gzip", chunks=True, maxshape=(None, 150, 96, 96))
        else:
            h5file["y_train"].resize(
                (h5file["y_train"].shape[0] + y_train.shape[0]), axis=0)
            h5file["y_train"][-y_train.shape[0]:] = y_train

        if not "y_test" in h5file:
            h5file.create_dataset("y_test", data=y_test,
                                  compression="gzip", chunks=True, maxshape=(None, 150, 96, 96))
        else:
            h5file["y_test"].resize(
                (h5file["y_test"].shape[0] + y_test.shape[0]), axis=0)
            h5file["y_test"][-y_test.shape[0]:] = y_test
        
        h5file.close()


if __name__ == '__main__':
    # print(load_label_csv())
    # load_npz()
    prepare_dataset()
    # append2h5py()