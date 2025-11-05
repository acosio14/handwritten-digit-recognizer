import torch
from torch import nn

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

    def train_loop(self, dataset, number_of_epochs, batch_size): # dataset = TensorDataset(X,y)
        for epoch in range(number_of_epochs): # start with 50 epochs
            total_loss = 0
            dataset_size = len(dataset)
            images, labels = dataset
            
            self.model.train()
            for i in range(dataset_size/batch_size):
                X_train = images[i:i+batch_size]
                y_train = labels[i:i+batch_size]
                
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
            print(f"Epoch {epoch + 1}")
            print(f"Train Loss: {total_loss / len(dataset)}")
            # print(f"Val Loss: {avg_val_loss}")
            print()
        # Create Validation step: Test predictions against validation data once its going higher stop.