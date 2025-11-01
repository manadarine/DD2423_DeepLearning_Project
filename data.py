import pickle
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data(file_path):
    with open(file_path, 'rb') as file:
        batch = pickle.load(file, encoding='bytes')
        images = batch[b'data'].astype(np.float32).T  # shape: (3072, n)
        labels = batch[b'labels']
        images = images / 255.0

        labels_onehot = np.zeros((10, len(labels)), dtype=np.float32)
        for i, label in enumerate(labels):
            labels_onehot[label, i] = 1.0

        return images, labels, labels_onehot

# def transform(images):
#     # normalize
#     d, n = images.shape
#     mean = np.mean(images, axis=1, keepdims=True)
#     std = np.std(images, axis=1, keepdims=True)
#     images = (images - mean) / std

#     # # reshape to (32, 32, 3, n)
#     # images = images.reshape((32, 32, 3, -1))
#     return torch.tensor(images, dtype=torch.float32)






def transform(images):
    # normalize
    d, n = images.shape
    mean = np.mean(images, axis=1, keepdims=True)
    std = np.std(images, axis=1, keepdims=True)
    images = (images - mean) / std

    # reshape to (n, 3, 32, 32)
    images = images.T.reshape(-1, 3, 32, 32)

    return torch.tensor(images, dtype=torch.float32)

def load_all_batches():
    print("Loading training and validation data...")

    images_list = []
    labels_list = []
    labels_onehot_list = []

    for i in range(1, 6):
        images, labels, labels_onehot = load_data(f'Datasets/cifar-10-batches-py/data_batch_{i}')
        images_list.append(images)
        labels_list.extend(labels)
        labels_onehot_list.append(labels_onehot)

    images = np.hstack(images_list)  # shape: (3072, 50000)
    labels = np.array(labels_list)
    labels_onehot = np.hstack(labels_onehot_list)

    train_images = transform(images[:, :-1000])
    train_labels = torch.tensor(labels[:-1000], dtype=torch.long)
    train_labels_onehot = torch.tensor(labels_onehot[:, :-1000], dtype=torch.float32)

    val_images = transform(images[:, -1000:])
    val_labels = torch.tensor(labels[-1000:], dtype=torch.long)
    val_labels_onehot = torch.tensor(labels_onehot[:, -1000:], dtype=torch.float32)

    return train_images, train_labels, train_labels_onehot, val_images, val_labels, val_labels_onehot

def load_test_data():
    images, labels, labels_onehot = load_data(f'Datasets/cifar-10-batches-py/test_batch')
    return transform(images), torch.tensor(labels, dtype=torch.long)
