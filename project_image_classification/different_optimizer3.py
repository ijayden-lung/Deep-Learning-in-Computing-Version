from __future__ import print_function, division

import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
print(torch.__version__)
from PIL import Image
import matplotlib.pyplot as plt
plt.ion()   # interactive mode

data_transforms = {
		    'train': transforms.Compose([
				     transforms.RandomHorizontalFlip(0.5),
					      transforms.ToTensor(),
						       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
							        
							       ]),
			    
			    'valid': transforms.Compose([
					     transforms.ToTensor(),
						      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
							      ]),
				    
				    'test': transforms.Compose([
						     transforms.ToTensor(),
							      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
								      ])
					}

# Load CIFAR10

image_datasets = {x: torchvision.datasets.CIFAR10(root='./data', train=(x=='train'), download=True, transform=data_transforms[x]) for x in ['train', 'valid','test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=(x=='train'), num_workers=4) for x in ['train', 'valid','test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid','test']}
class_names = image_datasets['train'].classes

# Move to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Plot the loss and accuracy curves for training and validation 
def loss_acc_plt(history):
    plt.rcParams['figure.figsize'] = (18, 8.0)
    fig, ax = plt.subplots(1,2)
    ax[0].plot(history['train_loss'], color='b', label="Training loss")
    ax[0].plot(history['valid_loss'], color='r', label="validation loss",axes =ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history['train_acc'], color='b', label="Training accuracy")
    ax[1].plot(history['valid_acc'], color='r',label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    
def train_model(model, criterion, optimizer, num_epochs=25, save_path='saved_weight.pth'):
    since = time.time()
    history = {}
    history['train_loss'] = []
    history['valid_loss'] = []
    history['train_acc'] = []
    history['valid_acc'] = []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train': model.train()  # Set model to training mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            #print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)
        
        for phase in ['valid']:
            if phase == 'valid':
                model.eval()   # Set model to evaluate mode

            running_valid_loss = 0.0
            running_valid_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
           
                with torch.no_grad():
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                # statistics
                running_valid_loss += loss.item() * inputs.size(0)
                running_valid_corrects += torch.sum(preds == labels.data)
            epoch_valid_loss = running_valid_loss / dataset_sizes[phase]
            epoch_valid_acc = running_valid_corrects.double() / dataset_sizes[phase]
            #print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_valid_loss, epoch_valid_acc))    
            history['valid_loss'].append(epoch_valid_loss)
            history['valid_acc'].append(epoch_valid_acc)
        
    print()

    time_elapsed = time.time() - since
    print('Last Epoch train Loss: {:.4f} Acc: {:.4f}'.format(history['train_loss'][-1], history['train_acc'][-1]))
    print('Last Epoch valid Loss: {:.4f} Acc: {:.4f}'.format(history['valid_loss'][-1], history['valid_acc'][-1]))
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    torch.save(model.state_dict(), save_path)
    return model,history

def test_model(model, load_path='saved_weight.pth'):    
    # load the model weights
    model.load_state_dict(torch.load(load_path))
    
    since = time.time()

    for phase in ['test']:
        if phase == 'test':
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
           

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

            # statistics
            running_corrects += torch.sum(preds == labels.data)
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print('{} Acc: {:.4f}'.format(phase, epoch_acc))

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return 

# Define a Convolutional Neural Network
class MyNet4(nn.Module):
    def __init__(self):
        super(MyNet4, self).__init__()
        # TODO Task 3 & 4: Design Your Network I & II 
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,32)
        self.fc4 = nn.Linear(32, 10)


    def forward(self, x):
        # TODO Task 3 & 4: Design Your Network I & II
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*6*6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    
model_ft = MyNet4() #Define the model
model_ft = model_ft.to(device)
print(model_ft)
# TODO Task 2:  Define loss criterion - cross entropy loss
criterion = nn.CrossEntropyLoss()
# TODO Task 2:  Define Optimizer
optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# TODO Task 2:  Train the model
(model,history) = train_model(model_ft, criterion, optimizer, num_epochs=25, save_path='saved_weight.mynet4_3drop')
# TODO Task 2:  Test the model
test_model(model_ft, load_path='saved_weight.mynet4_3drop')
