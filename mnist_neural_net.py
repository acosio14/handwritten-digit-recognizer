import torch
import os
from torch import nn
import matplotlib.pylab as plt
from datetime import datetime
from torchmetrics.classification import(
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassF1Score,
    MulticlassRecall,
)
import torch.nn.functional as F

class ImageNeuralNet(nn.Module):
    """Create a Feedforward Neural Network for an image.
    
    """
    def __init__(self,image_pixels):
        """Initialize an instance of ImageNeuralNet."""
        super().__init__()
        self.fc1 = nn.Linear(image_pixels,5)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(5,15)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(15,5)
        self.relu3 = nn.ReLU()
        self.output_layer = nn.Linear(5,10) 

    def forward(self, image):
        """Feedfoward architecture."""
        x = self.relu1(self.fc1(image))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.output_layer(x)

        return x

class ImageConvNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3,kernel_size=(3,3))
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.flatten1 = nn.Flatten()
        self.output_layer = nn.Linear(507,10) # 4 Channels x (26,26) + 4 (bias), (26,26) - no padding
    
    def forward(self,image):
        x = self.conv1(image)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.flatten1(x)
        x = self.output_layer(x)

        return x
    
class ModelTraining():
    """Create Training Loop for Neural Net Model."""
    def __init__(self, neural_network, optimzer, loss_function):
        """Initialize an instance of ModelTraining."""
        self.model = neural_network
        self.optimizer = optimzer
        self.loss_function = loss_function
        self.train_list = []
        self.validation_list = []
        self.best_val_loss = 10000
        self.best_model = None
        self.best_metrics = None

    def train_loop(self, training_set, validation_set, number_of_epochs, batch_size):
        """Train the Neural Net model and evaluate it."""
        accuracy = MulticlassAccuracy(num_classes=10).to(torch.device("mps"))
        precision = MulticlassPrecision(num_classes=10, average='weighted').to(torch.device("mps"))
        recall = MulticlassRecall(num_classes=10, average='weighted').to(torch.device("mps"))
        f1score = MulticlassF1Score(num_classes=10, average='weighted').to(torch.device("mps"))

        for epoch in range(number_of_epochs):

            # Training
            total_loss = 0
            dataset_size = len(training_set[0])
            images, labels = training_set
            self.model.train()
            for i in range(0, dataset_size, batch_size):
                if (i == dataset_size - 1) and (dataset_size % batch_size != 0):
                    batch_size = dataset_size % batch_size
                start = i
                end = i + batch_size

                X_train = images[start:end].to(torch.device("mps"))
                y_train = labels[start:end].to(torch.device("mps")).reshape(batch_size,)
                # Forward pass.
                logits = self.model(X_train)
                loss = self.loss_function(logits, y_train)
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            # Evaluation
            self.model.eval()
            val_set_size = len(validation_set[0])
            v_images, v_labels = validation_set
            v_total_loss = 0
            with torch.no_grad():
                for i in range(0, val_set_size, batch_size):
                    if (i == val_set_size - 1) and (val_set_size % batch_size != 0):
                        batch_size = val_set_size % batch_size
                    start = i
                    end = i + batch_size
                    X_val = v_images[start:end].to(torch.device("mps"))
                    y_val = v_labels[start:end].to(torch.device("mps")).reshape(batch_size,)
                    
                    logits = self.model(X_val)
                    probabilites = F.softmax(logits, dim=1)
                    y_pred = torch.argmax(probabilites, dim=1)

                    accuracy.update(y_pred, y_val)
                    precision.update(y_pred, y_val)
                    recall.update(y_pred, y_val)
                    f1score.update(y_pred, y_val)

                    vloss = self.loss_function(logits, y_val)
                    v_total_loss += vloss.item()
            
            average_train_loss = total_loss / (dataset_size/batch_size)
            average_val_loss = v_total_loss/ (val_set_size/batch_size)

            total_accuracy = accuracy.compute()
            total_precision = precision.compute()
            total_recall = recall.compute()
            total_f1score = f1score.compute()

            self.train_list.append(average_train_loss)
            self.validation_list.append(average_val_loss)

            if average_val_loss <= self.best_val_loss:
                self.best_val_loss = average_train_loss
                self.best_model = self.model
                self.best_metrics = (epoch, y_val, y_pred)

            print(f"Epoch {epoch + 1}")
            print(f"Train Loss: {self.train_list[-1]}")
            print(f"Val Loss: {self.validation_list[-1]}")

            print(f"Accuracy: {total_accuracy*100}")
            print(f"Precision: {total_precision*100}")
            print(f"Recall: {total_recall*100}")
            print(f"F1 Score: {total_f1score}")
            accuracy.reset()
            precision.reset()
            recall.reset()
            f1score.reset()
            print()

    def save_model(self, folder, filename, epochs, is_best=False):
        """Save the Neural Net model."""
        if is_best:
            epoch, *other = self.best_metrics
            saved_model = self.best_model.state_dict()
        else:
            epoch = epochs
            saved_model = self.model.state_dict()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cwd = os.getcwd()
        filename = '_'.join([filename,epoch])
        file_dir = os.path.join(cwd, folder)
        
        if not os.path.isdir(file_dir):
            print(f"Error: {file_dir} doesn't exists.")
            return
        
        new_file = os.path.join(file_dir, filename, timestamp,".pth")

        torch.save(saved_model, new_file)

    def plot_train_eval_figure(self):
        """Plot the train and validation curves from the Neural Net model."""
        if not self.train_list and not self.validation_list:
            print("Train or Evaluation List empty.")
        else:
            epochs = [*range(1,len(self.train_list) + 1)]
            plt.plot(epochs, self.train_list, label="Training", color="red")
            plt.plot(epochs, self.validation_list, label="Validation", color="blue")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()