# .idx1 - stores labels, with a header indicating the number of items and
# and then a sequence of single-byte labels

# .idx - stores image data, with a header indicating the number of images, height,
# and width, followed by the raw pixel data for each image.

import os
import matplotlib.pyplot as plt
import numpy as np

def read_idx(idx_file):
    with open(idx_file,'rb') as file:
        magic_number = file.read(4) #0 0 (data type) (num of dimensions fo stored arrays)
        dimension_1 = file.read(4) # 60000
        dimension_2 = file.read(4) # 28
        dimension_3 = file.read(4) # 28
        data = file.read()
    data_np = np.frombuffer(data, dtype=np.ubyte) #NEED datatype
    img_pixels = data_np.reshape(dimension_1,dimension_2,dimension_3)

def show_image(image_name: str,image_rgb: tuple[int, int, int]):

    plt.imshow(image_rgb)
    plt.title(image_name)
    plt.show()


def main():

    mnst_dataset = {
        'train_images': 'MNIST_dataset/train-images-idx3-ubyte/train-images-idx3-ubyte',
        'train_labels': 'MNIST_dataset/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
        'test_images': 'MNIST_dataset/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
        'test_labels': 'MNIST_dataset/t10k-images-idx1-ubyte/t10k-images-idx1-ubyte',
    }
    
    train_images_filepath = os.path.join(os.getcwd(),mnst_dataset.get('train_images'))

    train_image_data = read_idx(train_images_filepath)

if __name__ == "__main__":
    main()