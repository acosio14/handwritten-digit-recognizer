import torch
from torch import nn
import math

class ImageClassifier(nn.Module):
    def __init__(self,image_pixels): # image_pixel = 28*28 => 784
        super().__init__()
        self.fc1 = nn.Linear(image_pixels,512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512,128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128,64)
        self.relu3 = nn.ReLU()
        self.output_layer = nn.linear(64,1)

    def forward(self, image):
        x = self.relu1(self.fc1(image))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.output_layer(x)

        return x
    
class ModelTraining():
    def __init__(self, neural_network, optimzer, loss_function):
        self.model = neural_network
        self.optimizer = optimzer
        self.loss_function = loss_function


    def train_loop(self, train_set, val_set, number_of_epochs, batch_size):
        for epoch in range(number_of_epochs): # start with 50 epochs

            # Training
            total_loss = 0
            dataset_size = len(train_set)
            images, labels = train_set
            self.model.train()
            for i in range(0, dataset_size, batch_size):
                if (i == dataset_size - 1) and (dataset_size % batch_size != 0):
                    batch_size = dataset_size % batch_size
                start = i
                end = i + batch_size
                X_train = images[start:end]
                y_train = labels[start:end]
                # Forward pass.
                y_pred = self.model(X_train) # Makes prediction with X data
                loss = self.loss_function(y_pred, y_train) #Calculates Loss (y_pred - y_true)

                # Backward pass and optimization
                self.optimizer.zero_grad() # Reset the gradients of all optimized
                loss.backward() # Computes the gradeint of current tensor wrt graph leaves
                                # Traverses teh computational graph (built during forward pass)
                                # It calculates the gradients of the loss with respect to all tensors
                                # in the graph.
                self.optimizer.step() # Uses the gradients to updates the parameters, minimizing loss
                                      # and improving model's performance.

                total_loss += loss.item()

            # Evaluation
            self.model.eval()
            val_set_size = len(val_set)
            v_images, v_labels = val_set
            v_total_loss = 0
            with torch.no_grad():
                for i in range(0, val_set_size, batch_size):
                    if (i == val_set_size - 1) and (val_set_size % batch_size != 0):
                        batch_size = val_set_size % batch_size
                        X_val = v_images[start:end]
                        y_val = v_labels[start:end]
                        
                        y_pred = self.model(X_val)
                        loss = self.loss_function(y_pred, y_val)
                        v_total_loss += loss

            print(f"Epoch {epoch + 1}")
            print(f"Train Loss: {total_loss / dataset_size}")
            print(f"Val Loss: {v_total_loss/ val_set_size}")
            print()
        # Create Validation step: Test predictions against validation data once its going higher stop.

        # Train - model learn from this set
        # validation - used to evaluate models and tune them (dev set)
        # Test - used to compare different models and select best. Unbiased.