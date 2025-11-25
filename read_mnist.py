from typing import BinaryIO

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray


def read_imgages_idx(idx_file: BinaryIO) -> NDArray:
    """Process .idx file input images and convert to numpy array."""
    with open(idx_file, "rb") as file:
        magic_number = file.read(4)
        dimension_1 = int.from_bytes(file.read(4), byteorder="big", signed=False)
        dimension_2 = int.from_bytes(file.read(4), byteorder="big", signed=False)
        dimension_3 = int.from_bytes(file.read(4), byteorder="big", signed=False)
        data_np = np.frombuffer(file.read(), dtype=np.uint8)
    return data_np.reshape(dimension_1, dimension_2, dimension_3)


def read_labels_idx(idx_file: BinaryIO) -> NDArray:
    """Process .idx file labels and convert to numpy array."""
    with open(idx_file, "rb") as file:
        magic_number = file.read(4)
        dimension_1 = int.from_bytes(file.read(4), byteorder="big", signed=False)
        data_np = np.frombuffer(file.read(), dtype=np.uint8)
    return data_np.reshape(dimension_1, 1)


def show_image(image_name: str, gray_img: tuple[int, int, int]) -> None:
    """Plot image."""
    plt.imshow(gray_img)
    plt.axis("off")
    plt.title(image_name)
    plt.imshow(gray_img, cmap="gray")


def normalize_data(dataset: NDArray) -> NDArray:
    """Normalize data."""
    dataset_min = np.min(dataset)
    dataset_max = np.max(dataset)
    return (dataset - dataset_min) / (dataset_max - dataset_min)


def split(
    images_data: NDArray, labels: NDArray, val_ratio: float,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Split data into train and validation sets."""
    number_of_images = len(labels)
    shuffled_sequence = np.random.permutation(number_of_images)

    shuffled_images = images_data[shuffled_sequence]
    shuffled_labels = labels[shuffled_sequence]

    split_index = int((1 - val_ratio) * number_of_images)
    x_train = shuffled_images[:split_index]
    x_val = shuffled_images[split_index:]
    y_train = shuffled_labels[:split_index]
    y_val = shuffled_labels[split_index:]

    return x_train, x_val, y_train, y_val


def convert_numpy_to_flatten_tensor(numpy_array: NDArray) -> None:
    """Convert numpy to a flatten tensor."""
    tensor_array = torch.tensor(numpy_array, dtype=torch.float32)
    return torch.flatten(tensor_array, start_dim=1)
