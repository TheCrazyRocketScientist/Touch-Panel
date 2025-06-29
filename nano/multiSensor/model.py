import random
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from sklearn.manifold import TSNE


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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#import matplotlib
import matplotlib.pyplot as plt

#import counter class from collections
from collections import Counter


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
    def __init__(self, channels = 2, num_classes = 5, sequence_length=21, 
                 k1=3, k2=3,c1=16,c2=32,conv2_en=True,pool="max",dropout=0.50,
                 pool1_en = True,pool2_en=True,bn1_en=True,bn2_en=True):


        super().__init__()

        self.pool1_en = pool1_en
        self.pool2_en = pool2_en
        self.conv2_en = conv2_en
        self.bn1_en = bn1_en
        self.bn2_en = bn2_en

        #data format is (batch_size, channels, sequence_length)
        #output of 1D Conv Layer is given by: Lout = Lin-K+1

        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=c1, kernel_size=k1)
        self.conv1_output = sequence_length-k1+1

        if self.pool1_en or self.pool2_en:
            
            if pool == 'max':
                self.pool = nn.MaxPool1d(kernel_size=2)
            elif pool == 'avg':
                self.pool = nn.AvgPool1d(kernel_size=2)
            else: 
                self.pool = None

        
        if self.pool1_en:
            #if first pool is present output of conv1 is halved
            self.conv1_output = self.conv1_output//2
         
        #output of conv1 feeds into conv2
        self.conv2_input = self.conv1_output

        if self.conv2_en:

            self.conv2 = nn.Conv1d(in_channels=c1, out_channels=c2, kernel_size=k2)
            self.conv2_output = self.conv2_input-k2+1

            if self.pool2_en:
                #output of second pool halves the size of conv2
                self.conv2_output //= 2

            self.fc1 = nn.Linear(c2 * self.conv2_output, 64)
        
        else:

            self.fc1 = nn.Linear(c1*self.conv1_output,64)

        #last layer to output to final FC layer with 5 neurons, one for each class
        self.fc2 = nn.Linear(64, num_classes)

        #set a dropout layer with some % neurons dropped
        self.dropout = nn.Dropout(p=dropout)

        #set up batchnorm layers
        if self.bn1_en:
            self.bn1 = nn.BatchNorm1d(c1)
        if self.bn2_en:
            self.bn2 = nn.BatchNorm1d(c2)


    def forward(self, x, return_features=False):

        self.return_features = return_features

        # Input x shape: (batch_size, sequence_length, channels)
        x = x.permute(0, 2, 1)  # (batch, channels, sequence)

        x = self.conv1(x)

        if self.bn1_en:
            x = self.bn1(x)
        x = F.relu(x)

        if self.pool1_en:
            x = self.pool(x)

        if self.conv2_en:
            x = self.conv2(x)

            if self.bn2_en:
                x = self.bn2(x)

            x = F.relu(x)

            if self.pool2_en:
                x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        if self.return_features:
            return x

        x = self.dropout(x)
        x = self.fc2(x)
        return x


def handle_loss():
    pass


if __name__ == "__main__":

    #bool for cm or tsne plot: 
    tsne_plot = False

    #check GPU availibility

    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_epochs = 150
    batch_size = 64
    
    #get data of given window size and overlap
    data_version_one,labels_one = get_train_data(window_size=0.80,overlap=1.0,version=1)
    data_version_two,labels_two = get_train_data(window_size=0.80,overlap=1.0,version=2)

    data = np.concatenate([data_version_one,data_version_two],axis=0)
    labels = np.concatenate([labels_one,labels_two],axis=0)

    #get sequence length from shape of data
    sequence_length = data_version_one.shape[1]

    #get test train split
    X_train,X_test,y_train,y_test = train_test_split(data,labels,test_size=0.20,random_state=42,shuffle=True,stratify=labels)

    train_noise = X_train[y_train == 0]
    X_train = X_train[y_train != 0]
    y_train = y_train[y_train != 0]


    #specify how many noise samples will be introduced per epoch
    train_counts = np.unique_counts(y_train)
    epoch_noise_samples = int((train_counts.counts.sum())/(len(train_counts.counts)))

    #use custom dataset derived child class
    test_dataset = accelerometer_data(data=X_test,labels=y_test)

    #run on main thread and use Dataloader class for test and train split
    test_dataloader = DataLoader(dataset=test_dataset,shuffle=False,batch_size=batch_size,num_workers=0)

    #create an instance of tap cnn
    model = tap_cnn(channels=3, num_classes=5,sequence_length=sequence_length).to(device)

    #use adam not sgd
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    #use cross entropy as loss function
    criterion = nn.CrossEntropyLoss()

    #init all lists to generate graphs
    per_epoch_train_acc = []
    per_epoch_train_loss = []
    per_epoch_validation_acc = []
    per_epoch_validation_loss = []

    #training loop
    for epoch in range(num_epochs):

        #grab random noise samples and inject into training data per epoch
        noise_index_arr = np.random.choice(len(train_noise),size=epoch_noise_samples)
        epoch_noise = train_noise[noise_index_arr]

        #concat noise with training data
        epoch_data = np.concatenate([X_train,epoch_noise],axis=0)
        epoch_labels = np.concatenate([y_train,np.zeros(shape=(epoch_noise_samples,),dtype=int)],axis=0)

        #use dataset and dataloader to load train set with noise
        epoch_dataset = accelerometer_data(data=epoch_data,labels=epoch_labels)

        #drop_last was added to ensure uniform batch size per epoch
        epoch_dataloader = DataLoader(dataset=epoch_dataset,shuffle=True,drop_last=True,batch_size=batch_size,num_workers=0)

        #set model in training mode
        model.train()

        #initalize vars to log data per epoch
        epoch_loss = 0
        correct = 0
        total = 0

        #init vars to calculate validation stats
        val_loss = 0
        val_correct = 0
        val_total = 0

        #load data from epoch dataloader
        for data, labels in epoch_dataloader:

            #send data and labels to device
            data = data.to(device)
            labels = labels.to(device)

            #reset gradients
            optimizer.zero_grad()
            
            #get model output
            outputs = model(data)

            #calculate loss 
            loss = criterion(outputs, labels)

            #apply backprop
            loss.backward()
            optimizer.step()

            #accumulate total loss
            epoch_loss += loss.item() * batch_size

            #accumulate accuracy by checking logits
            #outputs are of the size (batch_size,number_of_classes)
            #use max along axis = 1

            _, predictions = torch.max(outputs, dim=1)
            total += batch_size
            correct += (predictions == labels).sum().item()

        #calculate avg loss and accuracy per epoch
        avg_loss = epoch_loss / total
        accuracy = 100 * correct / total

        #append data to lists
        per_epoch_train_acc.append(accuracy)
        per_epoch_train_loss.append(avg_loss)

        #run model in evaluation mode
        model.eval()

        #use no grad context manager to ensure no changes to gradients
        with torch.no_grad():

            #load data from test dataloader
            for data, labels in test_dataloader:

                #move data to device
                data = data.to(device)
                labels = labels.to(device)

                #get model output
                outputs = model(data)

                #calculate loss
                loss = criterion(outputs, labels)
                val_loss += loss.item() * data.size(0)

                #calculate number of correct predictions for accuracy by using logits
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        #calculate avg validation loss and accuracy
        val_avg_loss = val_loss/val_total
        val_accuracy = 100 * val_correct / val_total

        #append to lists
        per_epoch_validation_acc.append(val_accuracy)
        per_epoch_validation_loss.append(val_avg_loss)


        print(f"Epoch {epoch}/{num_epochs} Train Accuracy: {accuracy:.4f} Train Loss: {avg_loss:.4f} Validation Accuracy: {val_accuracy:.4f} Validation Loss: {val_avg_loss:.4f}")

# After the training loop, typically after the final print statement for epoch results


# Set model to evaluation mode
model.eval()

all_preds = []
all_labels = []
all_features = []

# Use no grad context manager
with torch.no_grad():
    for data, labels in test_dataloader:
        data = data.to(device)
        labels = labels.to(device)

        if tsne_plot == False:

            outputs = model(data,return_features=tsne_plot)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())

        else:
            features = model(data,return_features=tsne_plot)
            all_features.append(features.cpu().numpy())

        all_labels.extend(labels.cpu().numpy())

# Convert lists to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Generate confusion matrix or tsne plot

if tsne_plot == False:

    cm = confusion_matrix(all_labels, all_preds)

    class_names = ['Noise', 'Tap1', 'Tap2', 'Tap3', 'Tap4']

    # Display the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title('Confusion Matrix for Test Set')

else:
    all_features = np.concatenate(all_features,axis=0)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    tsne_results = tsne.fit_transform(all_features)

    # Plot
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        hue=all_labels,
        palette="tab10",
        alpha=0.8,
        s=60,
        edgecolor='k'
    )



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