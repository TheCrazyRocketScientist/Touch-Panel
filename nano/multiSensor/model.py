import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

#fix seed for reproducable results
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


#import get_train_data from label.py to get labelled windows
from label_data import get_train_data

#import test_train_split from sklearn
from sklearn.model_selection import train_test_split

#import matplotlib
import matplotlib.pyplot as plt


class accelerometer_data(Dataset):

    def __init__(self,data,labels,transform=None):

        super().__init__()
        self.data = torch.tensor(data=data,dtype=torch.float32)
        self.labels = torch.tensor(data=labels,dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        sample = self.data[index]
        label = self.labels[index]

        if self.transform:
            sample = self.transform(sample)
        return sample,label


class tap_cnn(nn.Module):
    def __init__(self, channels = 2, num_classes = 5):

        #21 Samples per window, Two channels
        #two conv layers, kernel size is 3
        #one max pool of size 2

        super().__init__()

        #data format is (batch_size, channels, sequence_length)
        #output of 1D Conv Layer is given by: Lout = Lin-K+1

        #input shape is (batch_size,2,21)
        #output of conv1 will be per sample from 21 to (21-3+1) = 19
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=16, kernel_size=3)

        #input shape is (batch_size,2,19)
        #output of max pool will be sample//2

        #after first max pool: (batch_size,2,9)
        #after second max pool: (batch_size,2,3)
        self.pool = nn.MaxPool1d(kernel_size=2)

        #after first max pool input to conv2 is in the shape (batch_size,2,9)
        #output of conv2 will be per sample from 9 to (9-3+1) = 7
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)

        #after second max pool input of 32*3 is given per batch to first FC Layer
        self.fc1 = nn.Linear(32 * 3, 64) 

        #last layer to output to final FC layer with 5 neurons, one for each class
        self.fc2 = nn.Linear(64, num_classes)

        #set a dropout layer with 35% neurons dropped
        self.dropout = nn.Dropout(p=0.35)


    def forward(self, x):

       # Input x shape: (batch_size, sequence_length, channels)
       # Convert to (batch_size, channels, sequence_length) for Conv1d
        x = x.permute(0,2,1)

        x = F.relu(self.conv1(x))  # Conv1d + ReLU
        x = self.pool(x)           # Max pooling #1
        x = F.relu(self.conv2(x))  # Conv1D + ReLU
        x = self.pool(x)           # Max pooling #2
        
        x = x.view(x.size(0), -1)  # Flatten for FC
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)            
        return x


if __name__ == "__main__":

    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_epochs = 50

    #partition and generate lables from accelerometers
    #uncomment for full run through entire dataset
    #my_dataset = accelerometer_data(*get_train_data())
    
    #setting up test train split
    data,labels = get_train_data()
    X_train,X_test,y_train,y_test = train_test_split(data,labels,test_size=0.30,random_state=42,shuffle=True,stratify=labels)

    #use custom dataset derived child class
    train_dataset = accelerometer_data(X_train,y_train)
    test_dataset = accelerometer_data(X_test,y_test)

    #run on main thread and use Dataloader class for test and train split
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=15,shuffle=True,num_workers=0)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=15,shuffle=False,num_workers=0)


    #create an instance of tap cnn
    model = tap_cnn(channels=2, num_classes=5).to(device)

    #use cross entropy as loss function
    criterion = nn.CrossEntropyLoss()

    #use adam not sgd
    optimizer = torch.optim.Adam(model.parameters(), lr=35e-4, weight_decay=4e-3)

    per_epoch_train_acc = []
    per_epoch_train_loss = []
    per_epoch_validation_acc = []
    per_epoch_validation_loss = []


    for epoch in range(num_epochs):

        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_num, (data, labels) in enumerate(train_dataloader, start=1):

            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * data.size(0)  # accumulate total loss
            _, predictions = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

        avg_loss = epoch_loss / total
        accuracy = 100 * correct / total

        per_epoch_train_acc.append(accuracy)
        per_epoch_train_loss.append(avg_loss)

        val_loss = 0
        val_correct = 0
        val_total = 0

        model.eval()
        with torch.no_grad():
            for data, labels in test_dataloader:
                data = data.to(device)
                labels = labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * data.size(0)
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_loss /= val_total
        val_accuracy = 100 * val_correct / val_total

        per_epoch_validation_acc.append(val_accuracy)
        per_epoch_validation_loss.append(val_loss)

    print(f"Epoch {epoch}/{num_epochs} Train Accuracy: {accuracy} Train Loss: {avg_loss} Validation Accuracy: {val_accuracy} Validation Loss: {val_loss}")


result_accuracy = pd.DataFrame({
    "train_accuracy": per_epoch_train_acc,
    "validation_accuracy": per_epoch_validation_acc
})
result_loss = pd.DataFrame({
    "train_loss": per_epoch_train_loss,
    "validation_loss": per_epoch_validation_loss,
})

result_accuracy.plot()
result_loss.plot()

plt.show()