import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BirdAudioClassifierModel(nn.Module): 
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



#model trainer class that should be used for training the model
class ModelTrainer: 
    def __init__(self, model, learning_rate): 
        #get the device (use the GPU) 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #move the model to the device
        self.model = model.to(self.device)

        #define the loss function 
        self.loss_func = torch.nn.CrossEntropyLoss()
        #define the optimizer 
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    def train(self, num_epochs, train_dataloader, test_dataloader): 
        #train the model 
    
        #keep track of the losses
        train_losses = []
        test_losses = []
        
        for t in range(num_epochs):
            print(f'Epoch {t+1}\n-------------------------------')
            #train the model 
            train_loss = self.train_epoch(train_dataloader=train_dataloader)
            train_losses.append(train_loss)
            #test the model 
            test_loss = self.test_epoch(test_dataloader=test_dataloader)
            test_losses.append(test_loss)
            
        print("Finished training!")

        return train_losses, test_losses

    def train_epoch(self, train_dataloader): 
        #set the model in training mode 
        self.model.train() 
    
        size = len(train_dataloader.dataset)
    
        #keep track of the total loss 
        total_loss = 0
    
        #loop over each batch
        for batch, (X, Y) in enumerate(train_dataloader): 
            #move all tensors to the device 
            X = X.to(self.device)
            Y = Y.to(self.device)
    
            #zero the gradient
            self.optimizer.zero_grad() 
            #get the model predictions 
            predictions = self.model(X)
            #compute the loss using the prediction and the label
            loss = self.loss_func(predictions, Y) 
    
            total_loss += loss
            
            #perform backpropagation 
            loss.backward() 
            self.optimizer.step() 
    
            #print progress
            if batch % 25 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')
    
        #calcualte the average loss
        avg_loss = total_loss / size
        return avg_loss.item()

    def test_epoch(self, test_dataloader): 
        #set the model in evaluation mode 
        self.model.eval() 
        #initialize the total test loss and the number of correct predictions 
        total_test_loss, correct_pred = 0, 0 
    
        #loop through the batch 
        with torch.no_grad(): 
            for batch, (X, Y) in enumerate(test_dataloader): 
                #move all tensors to the device 
                X = X.to(self.device)
                Y = Y.to(self.device)
                #predit classes 
                predictions = self.model(X) 
    
                #compute loss 
                total_test_loss += self.loss_func(predictions, Y).item() 
                correct_pred += (predictions.argmax(1)==Y).type(torch.float).sum().item()
    
        #comput the average loss and accuracy 
        avg_test_loss = total_test_loss / len(test_dataloader.dataset) 
        avg_correct_pred = correct_pred / len(test_dataloader.dataset) 
    
        print(f'\nTest Error:\nacc: {(100*avg_correct_pred):>0.1f}%, avg loss: {avg_test_loss:>8f}\n')
        return avg_test_loss


    def save_model(self, path): 
        torch.save(self.model.state_dict(), path)
        print(f"Saved model at: {path}")

    def load_model(self, path): 
        self.model.load_state_dict(torch.load(path))
        print(f"Loaded model from: {path}")
        
        
