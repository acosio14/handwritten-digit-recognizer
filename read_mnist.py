# .idx1 - stores labels, with a header indicating the number of items and
# and then a sequence of single-byte labels

# .idx - stores image data, with a header indicating the number of images, height,
# and width, followed by the raw pixel data for each image.

import os
import matplotlib.pyplot as plt
import numpy as np

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

def standardize_data(dataset):
    # Standardization of dataset
    data_mean = np.mean(dataset)
    data_std = np.std(dataset)

    return (dataset - data_mean) / data_std

def main():

    mnst_dataset = {
        'train_images': 'MNIST_dataset/train-images-idx3-ubyte/train-images-idx3-ubyte',
        'train_labels': 'MNIST_dataset/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
        'test_images': 'MNIST_dataset/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
        'test_labels': 'MNIST_dataset/t10k-images-idx1-ubyte/t10k-images-idx1-ubyte',
    }
    
    train_images_filepath = os.path.join(os.getcwd(),mnst_dataset.get('train_images'))
    train_labels_filepath = os.path.join(os.getcwd(),mnst_dataset.get('train_labels'))
    test_images_filepath = os.path.join(os.getcwd(),mnst_dataset.get('test_images'))
    test_labels_filepath = os.path.join(os.getcwd(),mnst_dataset.get('test_labels'))

    train_images = read_imgages_idx(train_images_filepath)
    train_labels = read_labels_idx(train_labels_filepath)
    test_images = read_imgages_idx(test_images_filepath)
    train_labels = read_labels_idx(test_labels_filepath)

    show_image(train_images)

if __name__ == "__main__":
    main()