import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

class ClassifierNetwork(nn.Module):
    """
    A Convolutional Neural Network (CNN) for image classification.

    :param IMAGE_SIZE: The height and width of the input images.
    :type IMAGE_SIZE: tuple

    Attributes:
        input_size (tuple): The input image size.
        conv1, conv2, conv3, conv4 (nn.Conv2d): Convolutional layers.
        bn1, bn2, bn3, bn4 (nn.BatchNorm2d): Batch normalization layers.
        fc1, fc2 (nn.Linear): Fully connected layers.
        _flattened_size (int): The flattened size after the convolutional layers.
    """

    def __init__(self, IMAGE_SIZE):
        """
        Initializes the ClassifierNetwork with the given image size.

        :param IMAGE_SIZE: The height and width of the input images.
        :type IMAGE_SIZE: tuple
        """
        super(ClassifierNetwork, self).__init__()
        self.input_size = IMAGE_SIZE
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2, stride=2)
        self.bn1 = nn.BatchNorm2d(16)

        # Second convolutional block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(32)

        # Third convolutional block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=3) 
        self.bn3 = nn.BatchNorm2d(64)

        # Fourth convolutional block
        self.conv4 = nn.Conv2d(64, 128, kernel_size=4, padding=2, stride=3)  
        self.bn4 = nn.BatchNorm2d(128)

        # Calculate the flattened size after convolutions
        self._flattened_size = self._compute_flattened_size(self.input_size)

        # Fully connected layers
        self.fc1 = nn.Linear(self._flattened_size, 512)
        self.fc2 = nn.Linear(512, 2)

    def _compute_flattened_size(self, input_size):
        """
        Computes the flattened size of the input after passing through all convolutional layers.

        :param input_size: The height and width of the input image.
        :type input_size: tuple

        :return: The number of elements after flattening the output from the convolutional layers.
        :rtype: int
        """
        x = torch.zeros(1, 3, *input_size)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x.numel()

    def forward(self, x):
        """
        Defines the forward pass of the model.

        :param x: The input tensor of images.
        :type x: torch.Tensor

        :return: The output tensor after passing through the CNN layers and fully connected layers.
        :rtype: torch.Tensor
        """
        # Convolutional layers with ReLU activation
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class Model():
    """
    A wrapper for training, validating, and testing a neural network model.

    :param network: The neural network to train.
    :type network: nn.Module

    :param epochs: The number of epochs to train the model.
    :type epochs: int

    Attributes:
        device (torch.device): The device on which to run the model ('cuda' or 'cpu').
        model (nn.Module): The neural network model.
        criterion (nn.Module): The loss function used during training.
        optimizer (torch.optim.Optimizer): The optimizer used during training.
        scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): The learning rate scheduler.
        num_epochs (int): The number of epochs to train for.
    """

    def __init__(self, network, epochs):
        """
        Initializes the Model with the given network and number of epochs.

        :param network: The neural network to train.
        :type network: nn.Module

        :param epochs: The number of epochs to train the model.
        :type epochs: int
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = network.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.02)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=2)
        self.num_epochs = epochs

    def train(self, train_loader):
        """
        Trains the model on the given training data.

        :param train_loader: The data loader for the training dataset.
        :type train_loader: DataLoader
        """
        prev_loss = float('inf')
        worse_loss_counter = 0
        
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            batches = 0
            for inputs, labels, paths in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                print(f"Batches: {batches}", end="\r")
                batches += 1
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {running_loss/len(train_loader)}")

            # Early stopping checking
            if running_loss > prev_loss:
                worse_loss_counter += 1
                if worse_loss_counter > 3:  # patience
                    print("Early stopping, results are not improving fast enough")
                    break
            
            prev_loss = running_loss

    def predict(self, test_loader):
        """
        Makes predictions on the test data and computes the accuracy.

        :param test_loader: The data loader for the test dataset.
        :type test_loader: DataLoader

        :return: A dictionary with the image paths as keys and the true and predicted labels as values.
        :rtype: dict
        """
        results = {}
        
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels, paths in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Saving info in results
                for path, true_label, pred_label in zip(paths, labels.cpu().numpy(), predicted.cpu().numpy()):
                    results[path] = [true_label, pred_label]
        
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy}%")
        return results
