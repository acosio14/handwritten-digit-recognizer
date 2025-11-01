# .idx1 - stores labels, with a header indicating the number of items and
# and then a sequence of single-byte labels

# .idx - stores image data, with a header indicating the number of images, height,
# and width, followed by the raw pixel data for each image.

import os

def read_ubyte(path):
    ...


def main():

    mnst_dataset = {
        'train_images': 'MNIST_dataset/train-images-idx3-ubyte/train-images-idx3-ubyte',
        'train_labels': 'MNIST_dataset/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
        'test_images': 'MNIST_dataset/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
        'test_labels': 'MNIST_dataset/t10k-images-idx1-ubyte/t10k-images-idx1-ubyte',
    }
    
    train_images_path = os.path.join(os.getcwd(),mnst_dataset.get('train_images'))

    train_image_data = read_ubyte(train_images_path)

if __name__ == "__main__":
    main()