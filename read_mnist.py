# .idx1 - stores labels, with a header indicating the number of items and
# and then a sequence of single-byte labels

# .idx - stores image data, with a header indicating the number of images, height,
# and width, followed by the raw pixel data for each image.

import os
import matplotlib.pyplot as plt
import numpy as np
import torch

def read_imgages_idx(idx_file):
    with open(idx_file,'rb') as file:
        magic_number = file.read(4) #0 0 (data type) (num of dimensions fo stored arrays)
        dimension_1 = int.from_bytes(file.read(4), byteorder='big', signed=False) # 60000
        dimension_2 = int.from_bytes(file.read(4), byteorder='big', signed=False)# 28
        dimension_3 = int.from_bytes(file.read(4), byteorder='big', signed=False)# 28
        data_np = np.frombuffer(file.read(), dtype=np.uint8) #grayscale (8-bit unsigned integer)
    return data_np.reshape(dimension_1,dimension_2,dimension_3) #img pixels (img, row, column)

def read_labels_idx(idx_file):
    with open(idx_file,'rb') as file:
        magic_number = file.read(4) #0 0 (data type) (num of dimensions fo stored arrays)
        dimension_1 = int.from_bytes(file.read(4), byteorder='big', signed=False)
        data_np = np.frombuffer(file.read(), dtype=np.uint8) 
    return data_np.reshape(dimension_1,1) 

def show_image(image_name: str,gray_img: tuple[int, int, int]):

    plt.imshow(gray_img)
    plt.axis('off')
    plt.title(image_name)
    plt.imshow(gray_img,cmap='gray')

def standardize_data(dataset): # Might need a reverse standardize. To get back to original data
    # Standardization of dataset
    print(f"Standardized with mean:{np.mean(dataset)} and std:{np.std(dataset)}")
    return (dataset - np.mean(dataset)) / np.std(dataset)

def split(images_data, labels, val_ratio):

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

def convert_numpy_to_flatten_tensor(numpy_array):
    tensor_array = torch.tensor(numpy_array)
    return torch.flatten(tensor_array)


def main():

    mnst_dataset = {
        'train_images': 'MNIST_dataset/train-images-idx3-ubyte/train-images-idx3-ubyte',
        'train_labels': 'MNIST_dataset/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
        'test_images': 'MNIST_dataset/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
        'test_labels': 'MNIST_dataset/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte',
    }
    
    train_images_filepath = os.path.join(os.getcwd(),mnst_dataset.get('train_images'))
    train_labels_filepath = os.path.join(os.getcwd(),mnst_dataset.get('train_labels'))
    test_images_filepath = os.path.join(os.getcwd(),mnst_dataset.get('test_images'))
    test_labels_filepath = os.path.join(os.getcwd(),mnst_dataset.get('test_labels'))

    train_images = read_imgages_idx(train_images_filepath)
    train_labels = read_labels_idx(train_labels_filepath)
    test_images = read_imgages_idx(test_images_filepath)
    test_labels = read_labels_idx(test_labels_filepath)

    X_train, X_val, y_train, y_val = split(train_images, train_labels, 0.2)

    X_train = standardize_data(X_train)
    X_val = standardize_data(X_val)
    
    # Need to flatten X_train before inputting in NN
    # Need to train a NN: image pixels are my features -> prediction, 
    # image_label is my target. Loss is pred_imag - true_label
    # At first, predicted value can be random but with more iteration/training it will "learn"
    # to match the labels.

if __name__ == "__main__":
    main()