import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BirdAudioClassifier(nn.Module): 
    def __init__(self, num_classes=88): 
        super().__init__() 
        #define the different layers 
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) #number of input channels, number of output channels (number of filters that will be applied), 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        #define max pooling 
        self.max_pool = nn.MaxPool2d(2, 2)
        #layer to flatten the output 
        self.flatten = nn.Flatten() 
        #fully connected layer for classification 
        self.fc1 = nn.Linear(411648, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    #implement the forward propagation through the network
    def forward(self, x): 
        #first convolutional layer + relu + max pool layer
        x = self.max_pool(F.relu(self.conv1(x)))
        #second convolutional layer + relu + max pool layer
        x = self.max_pool(F.relu(self.conv2(x)))
        #third convolutional layer + relu + max pool layer
        x = self.max_pool(F.relu(self.conv3(x)))
        #flatten the output from the convolutional layer 
        x = self.flatten(x) 
        #first fully connected layer 
        x = F.relu(self.fc1(x)) 
        #second fully connected layer 
        x = self.fc2(x) 
        #softmax activation function to convert the logits to log probabilities
        x = F.log_softmax(x, dim=1)
        return x