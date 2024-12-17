import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

class ClassifierNetwork(nn.Module):
    def __init__(self, IMAGE_SIZE):
        super(ClassifierNetwork, self).__init__()
        self.input_size = IMAGE_SIZE
        
        #first convolutional block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2, stride=2)
        self.bn1 = nn.BatchNorm2d(16)

        #second convolutional block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(32)

        #third convolutional block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=3) 
        self.bn3 = nn.BatchNorm2d(64)

        #fourth convolutional block
        self.conv4 = nn.Conv2d(64, 128, kernel_size=4, padding=2, stride=3)  
        self.bn4 = nn.BatchNorm2d(128)

        #calculate the flattened size after convolutions
        self._flattened_size = self._compute_flattened_size(self.input_size)

        #fully connected layers
        self.fc1 = nn.Linear(self._flattened_size, 512)
        self.fc2 = nn.Linear(512, 2)

    #this to compute the output size after the cnn layers
    def _compute_flattened_size(self, input_size):
        x = torch.zeros(1, 3, *input_size)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x.numel()

    def forward(self, x):        
        #conv layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        #flatten
        x = x.view(x.size(0), -1)

        #fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class Model():
    def __init__(self, network, epochs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = network.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.02)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=2)
        self.num_epochs=epochs

    def train(self,train_loader):   
        #early stopping variables
        prev_loss=float('inf')
        worse_loss_counter=0
        
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            batches=0
            for inputs, labels, paths in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                print(f"Batches: {batches}", end="\r")
                batches+=1
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                

            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {running_loss/len(train_loader)}")

            #early stopping checking
            if (running_loss>prev_loss):
                worse_loss_counter+=1
                if (worse_loss_counter>3): #patience
                    print("Early stopping, results are not improving fast enough")
                    break;
            
            prev_loss=running_loss
                        

    def predict (self, test_loader):
        #dictionary for storing results
        results={}
        
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

                #saving info in results
                for path, true_label, pred_label in zip(paths, labels.cpu().numpy(), predicted.cpu().numpy()):
                    results[path]=[true_label, pred_label]
        
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy}%")
        return results
   
