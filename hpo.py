#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import logging
import argparse
import logging
import os
import sys

# Setting up some basic configs for enabling logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))





def test(model, test_loader, criterion):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
     # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set model to evaluation mode
    model.eval()

    # Initialize variables for calculating accuracy
    running_corrects = 0
    running_loss = 0
    

    # Disable gradient calculation
    with torch.no_grad():
        # Loop over the test data
        for inputs, targets in test_loader:
            # Move inputs and targets to device
            inputs=inputs.to(device)
            targets=targets.to(device)
            outputs=model(inputs)
            loss=criterion(outputs, targets)
            pred = outputs.argmax(dim=1, keepdim=True)
            running_loss += loss.item() * inputs.size(0) #calculate running loss
            running_corrects += pred.eq(targets.view_as(pred)).sum().item() #calculate running corrects

        total_loss = running_loss / len(test_loader.dataset)
        total_acc = running_corrects/ len(test_loader.dataset)
        print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            total_loss, running_corrects, len(test_loader.dataset), 100.0 * total_acc
        ))
            


def train(model, train_loader, validation_loader, criterion, optimizer):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(2):
        
        model.train()
    
        running_loss = 0.0
        running_corrects = 0
        running_samples = 0
    
        # Loop over the training data
        for inputs, targets in train_loader:
            # Move inputs and targets to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item() * inputs.size(0)
            running_samples+=len(inputs)
            
            #NOTE: Comment lines below to train and test on whole dataset
            if (running_samples>(0.1* len(train_loader.dataset))):
                break

        

        # Evaluate on validation data
        model.eval()

        validation_loss = 0.0
        validation_corrects = 0
        validation_samples = 0

        with torch.no_grad():
            for inputs, targets in validation_loader:
                # Move inputs and targets to device
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
           
                pred = outputs.argmax(dim=1, keepdim=True)
                validation_loss += loss.item() * inputs.size(0) #calculate running loss
                validation_corrects += pred.eq(targets.view_as(pred)).sum().item() #calculate running corrects
                validation_samples+=len(inputs)
                
                if (validation_samples>(0.1*len(validation_loader.dataset))):
                    break
                
        total_loss = validation_loss / len(validation_loader.dataset)
        total_acc =  validation_corrects/ len(validation_loader.dataset)
        print("\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
         total_loss, validation_corrects, len(validation_loader.dataset), 100.0 * total_acc
        )) 
        
        return model
                
      
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    resnet50 = models.resnet50(pretrained=True)
    for param in resnet50.parameters():
        param.requires_grad = False

    resnet50.fc = nn.Sequential(nn.Linear(resnet50.fc.in_features, 256),
                                nn.ReLU(inplace = True),
                                nn.Linear(256, 133),
                                nn.ReLU(inplace = True))

    return resnet50

def create_data_loaders(data_path, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_data_path = os.path.join(data_path , 'train') # Calling OS Environment variable and split it into 3 sets
    test_data_path = os.path.join(data_path, 'test')
    validation_data_path=os.path.join(data_path, 'valid')
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ]) # transforming the training image data
                                                            
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ]) # transforming the testing image data
    
   
    # loading train,test & validation data from S3 location using torchvision datasets' Imagefolder function
    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size) 
    
    
    return train_data_loader, test_data_loader, validation_data_loader 

def main(args):
    logging.debug(args)
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss(ignore_index=133)
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader, test_loader, validation_loader=create_data_loaders(args.data_path, args.batch_size)
    
    model=train(model, train_loader, validation_loader, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
 
   
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument('--data_path', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    
    args = parser.parse_args()
    
    
    main(args)
