# Handwritten Digit Recognizer

This is my second AI project. The goal was to develop a Convolutional Neural Network (CNN) to detect handwritten digits using the classic MNIST dataset. I also built a Feedforward Neural Network (FNN) to compare its performance against the CNN. The MNIST dataset contains about 70,000 images, split into 60,000 training examples and 10,000 test examples.

For preprocessing, the original training set was further divided into training and validation subsets, with 20% reserved for validation. All images were then normalized to help the models learn effectively. Both networks were trained using PyTorch’s `CrossEntropyLoss` and the ADAM optimizer with a learning rate of 0.003. Training was done in batches of 32 images for 10 epochs.

To evaluate performance, the final models were tested on the untouched test dataset. Metrics such as accuracy, precision, recall, and F1-score were used to compare the networks. Overall, the CNN outperformed the FNN, which was to be expected since it is better suited to deal with spatial data, such as images. The entire workflow, including preprocessing, training, and evaluation can be viewed in the Jupyter notebook [handwritten_digits](https://github.com/acosio14/handwritten-digit-recognizer/blob/main/handwritten_digits.ipynb).

Overall, I learned:
- How to build a Convolutional Neural Network  
- How to properly batch data from scratch  
- A deeper understanding of classification metrics  
- The correct use of `CrossEntropyLoss` and the type of inputs it requires  
