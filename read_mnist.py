import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import BinaryIO, Tuple
from numpy.typing import NDArray

def read_imgages_idx(idx_file: BinaryIO) -> NDArray:
    """Process .idx file input images and convert to numpy array."""
    with open(idx_file,'rb') as file:
        magic_number = file.read(4)
        dimension_1 = int.from_bytes(file.read(4), byteorder='big', signed=False)
        dimension_2 = int.from_bytes(file.read(4), byteorder='big', signed=False)
        dimension_3 = int.from_bytes(file.read(4), byteorder='big', signed=False)
        data_np = np.frombuffer(file.read(), dtype=np.uint8)     # grayscale (8-bit unsigned integer)
    return data_np.reshape(dimension_1,dimension_2,dimension_3)  # img pixels (img, row, column)

def read_labels_idx(idx_file: BinaryIO) -> NDArray:
    """Process .idx file labels and convert to numpy array."""
    with open(idx_file,'rb') as file:
        magic_number = file.read(4)
        dimension_1 = int.from_bytes(file.read(4), byteorder='big', signed=False)
        data_np = np.frombuffer(file.read(), dtype=np.uint8) 
    return data_np.reshape(dimension_1,1) 

def show_image(image_name: str, gray_img: Tuple[int, int, int]) -> None:
    """Plot image."""
    plt.imshow(gray_img)
    plt.axis('off')
    plt.title(image_name)
    plt.imshow(gray_img,cmap='gray')

def standardize_data(dataset: NDArray) -> NDArray: 
    """Standardize data."""
    print(f"Standardized with mean: {np.round(np.mean(dataset),2)} and std: {np.round(np.std(dataset),2)}")
    return (dataset - np.mean(dataset)) / np.std(dataset)

def normalize_data(dataset: NDArray) -> NDArray:
    """Normalize data."""
    dataset_min = np.min(dataset)
    dataset_max = np.max(dataset)
    return (dataset - dataset_min) / (dataset_max - dataset_min)

def split(images_data: NDArray, labels: NDArray, val_ratio: float) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Split data into train and validation sets."""

    number_of_images = len(labels)
    shuffled_sequence = np.random.permutation(number_of_images)

    shuffled_images = images_data[shuffled_sequence]
    shuffled_labels = labels[shuffled_sequence]
    
    split_index = int((1 - val_ratio) * number_of_images)
    X_train = shuffled_images[:split_index]
    X_val = shuffled_images[split_index:]
    y_train = shuffled_labels[:split_index]
    y_val = shuffled_labels[split_index:]

    return X_train, X_val, y_train, y_val

def convert_numpy_to_flatten_tensor(numpy_array: NDArray):
    """Convert numpy to a flatten tensor."""
    tensor_array = torch.tensor(numpy_array,dtype=torch.float32)
    return torch.flatten(tensor_array, start_dim=1)


def main():
    ...

if __name__ == "__main__":
    main()