# -*- coding: utf-8 -*-
"""
Transfer Learning for Computer Vision Tutorial
==============================================
**Author**: `Sasank Chilamkurthy <https://chsasank.github.io>`_

In this tutorial, you will learn how to train a convolutional neural network for
image classification using transfer learning. You can read more about the transfer
learning at `cs231n notes <https://cs231n.github.io/transfer-learning/>`__

"""
# License: BSD
# Author: Sasank Chilamkurthy
# Modify for PML project : Al-Dam / Liebgott

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from tqdm import tqdm
from torch.optim import Adam

cudnn.benchmark = True
plt.ion()   # interactive mode

######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

######################################################################
# Training the model
# ------------------

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict()) # copy le model 
    best_acc = 0.0
    
    # variables to generate the accuracy curve 
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    #error_values = []

    for epoch in tqdm(range(num_epochs), desc='Epochs'): # using tqsm to setup a progrssion bar and see how fast go the training
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
                print('Training...')
            else:
                model.eval()   
                print('Validating...')

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase.capitalize()} iteration', leave=False):
                inputs = inputs.to(device) #transfer the device to the gpu 
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
                        loss.backward() #backward propagation 
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == 'train':
                scheduler.step()    
                train_losses.append(loss.item())
                train_accs.append(epoch_acc.item())
            else:
                val_losses.append(loss.item())
                val_accs.append(epoch_acc.item())
                #error_values.append(1 - loss.item())  


            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,train_losses,train_accs,val_losses,val_accs


######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

######################################################################
# Save the train model 
# ^^^^^^^^^^^^^^^^^^^^
#

def save_model(model_ft,train_name):
    
    model_folder = r'output\model'
    
    folder_is_existing = True
    index = 0
    
    while folder_is_existing:
        name_folder = f'model_{train_name}' #adapt name
        if index > 0:
            name_folder += f'_{index}'
        model_path = os.path.join(model_folder, name_folder)
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
            folder_is_existing = False
        else:
            #add +1 in the name if the folder already exist 
            index += 1
    
    #save the model in the correct file 
    
    model_file_name = os.path.join(model_path, 'model.pth')
    print("model saved")
    torch.save(model_ft.state_dict(), model_file_name)

######################################################################
# Save the confusion matrix 
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#

def confusion_matrix_generate_val(model,data_dir,cf_name):
    
    print("Generation val CM")
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                        data_transforms[x])
                for x in ['val']}
    # create image in a loaders 
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['val']}
    
    true_labels = []
    predicted_labels = []
    model.eval()

    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        predicted_labels.extend(preds) 

        labels = labels.data.cpu().numpy() # convert the labels to Numpy tabel able to use operation on it
        true_labels.extend(labels)

    classes = ('AKIEDC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC')
    
    # convert the predicted and true Label list into tensor
    predicted_labels = torch.tensor(predicted_labels)
    true_labels = torch.tensor(true_labels)

    # convert the predicted and true Label list into Numpy tab
    predicted_labels = predicted_labels.cpu().numpy()
    true_labels = true_labels.cpu().numpy()

    cm = confusion_matrix(true_labels, predicted_labels)

    df = pd.DataFrame(cm / np.sum(cm, axis=1)[:, None], index=classes, columns=classes) #confusion matrix is normalized each output is divised by sum of the line to get propotion
    # index get each ligne which class is it 
    # columns like index for columns 
    
    # figure size 
    plt.figure(figsize=(12, 7))

    axes = sn.heatmap(df, annot=True)  #get heatmap to see it better
    axes.set_title("val Confusion Matrix")
    axes.set_xlabel("Predicted labels")
    axes.set_ylabel("True labels")
    
    cm_folder_save = r'output\conf_matrix_val'

    #save the output
    
    folder_is_existing = True
    index = 0
    while folder_is_existing:
        name_folder = f'model_{cf_name}'
        if index > 0:
            name_folder += f'_{index}'
        cm_path = os.path.join(cm_folder_save, name_folder)
        if not os.path.exists(cm_path):
            os.makedirs(cm_path, exist_ok=True)
            folder_is_existing = False
        else:
            index += 1 
     
    plt.savefig(os.path.join(cm_path, 'output.png'))  
    print("conf metrics save")
    plt.title("val Confusion Matrix")
    plt.show()

def confusion_matrix_generate_train(model,data_dir,cf_name):
    
    print("Generation train CM")
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                        data_transforms[x])
                for x in ['train']}
    # create image in a loaders 
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                shuffle=True, num_workers=4)
                for x in ['train']}
    
    true_labels = []
    predicted_labels = []
    model.eval()

    for inputs, labels in dataloaders['train']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        predicted_labels.extend(preds)

        labels = labels.data.cpu().numpy()
        true_labels.extend(labels)

    classes = ('AKIEDC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC')
    
    predicted_labels = torch.tensor(predicted_labels)
    true_labels = torch.tensor(true_labels)

    predicted_labels = predicted_labels.cpu().numpy()
    true_labels = true_labels.cpu().numpy()

    cm = confusion_matrix(true_labels, predicted_labels)

    df = pd.DataFrame(cm / np.sum(cm, axis=1)[:, None], index=classes, columns=classes)

    #figure size 
    plt.figure(figsize=(12, 7))
    
    axes = sn.heatmap(df, annot=True)
    axes.set_title("Train Confusion Matrix")
    axes.set_xlabel("Predicted labels")
    axes.set_ylabel("True labels")


    cm_folder_save = r'output\conf_matrix_train'

    #save the output
     
    folder_is_existing = True
    index = 0
    while folder_is_existing:
        name_folder = f'model_{cf_name}'
        if index > 0:
            name_folder += f'_{index}'
        cm_path = os.path.join(cm_folder_save, name_folder)
        if not os.path.exists(cm_path):
            os.makedirs(cm_path, exist_ok=True)
            folder_is_existing = False
        else:
            index += 1  
    
    plt.savefig(os.path.join(cm_path, 'output.png'))  
    print("conf metrics save")
    plt.title("train Confusion Matrix")
    plt.show()

def confusion_matrix_generate_test(model_ft,data_dir,cf_name):
     
    print("Generation test CM")
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                        data_transforms[x])
                for x in ['test']}
    # create image in a loaders 
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['test']}
    
    true_labels = []
    predicted_labels = []
    model_ft.eval()
    
    # collect the true/false prediction 
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)
        predicted_labels.extend(preds)  

        labels = labels.data.cpu().numpy()
        true_labels.extend(labels) 
        
    classes = ('AKIEDC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC')

    predicted_labels = torch.tensor(predicted_labels)
    true_labels = torch.tensor(true_labels)

    predicted_labels = predicted_labels.cpu().numpy()
    true_labels = true_labels.cpu().numpy()
    
    cm = confusion_matrix(true_labels, predicted_labels)
    
    df = pd.DataFrame(cm / np.sum(cm, axis=1)[:, None], index=classes, columns=classes)
    
    #figure size 
    plt.figure(figsize=(12, 7))
    
    axes = sn.heatmap(df, annot=True)
    axes.set_title("Test Confusion Matrix")
    axes.set_xlabel("Predicted labels")
    axes.set_ylabel("True labels")
    
    cm_folder = r'output\conf_matrix'

    # Verify if the confusionmaxtrix folder already exist or not 
    folder_is_existing = True
    index = 0
    while folder_is_existing:
        name_folder = f'model_{cf_name}'
        if index > 0:
            name_folder += f'_{index}'
        cm_path = os.path.join(cm_folder, name_folder)
        if not os.path.exists(cm_path):
            os.makedirs(cm_path, exist_ok=True)
            folder_is_existing = False
        else:
            #add +1 if the folder already exist 
            index += 1 
    #save the output 
    plt.savefig(os.path.join(cm_path, 'output.png'))  
    print("conf metrics save")
    plt.title("test Confusion Matrix")
    plt.show()


######################################################################
# Generate accuracy curve  
# ^^^^^^^^^^^^^^^^^^^^^^^
#

def accuracy_curve(train_losses,train_accs,val_losses,val_accs,accuracy_curve_name):
    output_folder = r'output\accuracy_curve'
    os.makedirs(output_folder, exist_ok=True)

    # create file based on 'accuracy_curve_name'
    filename = accuracy_curve_name.split('.')[0] + '_accuracy_curve.png'
    filepath = os.path.join(output_folder, filename)
    
    # Generate x axes with epoch
    epochs = range(1, len(train_losses) + 1)

    # Trace the accuracy curve
    plt.figure()
    plt.plot(epochs,train_losses, label='Train Loss')
    plt.plot(epochs,train_accs, label='Train Accuracy')
    plt.plot(epochs,val_losses, label='Validation Loss')
    plt.plot(epochs,val_accs, label='Validation Accuracy')
    #plt.plot(epochs, error_values, label='Error Value')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)

    # save the curve in file 
    plt.savefig(filepath)
    plt.show()
    
if __name__ == '__main__':
# Data augmentation and normalization for training
# Just normalization for validation

    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(450, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), #conv en tensort pour le modele 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((450, 600)),
        transforms.CenterCrop(450),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((450, 600)),
        transforms.CenterCrop(450),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }
    
    ######################################################################
    # Setup the train data
    # --------------------
    #
    
    data_dir = r'dx7_imageRichtigVerteiltBlend'
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #sure graphic card 
    print(device)

    # get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])

    model_ft = models.resnet50(pretrained=True) #nehmt das model
    num_ftrs = model_ft.fc.in_features #in_feature eingang auf unsere schicht 

    # last layer 
    
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
  
    model_ft.fc = nn.Linear(num_ftrs, 7) #type de übetragungfuncktion #######################anderung 
    
    # model_ft.fc = nn.Sequential( # to create linear sequence layer 
    # nn.Linear(num_ftrs, 256), #adding a linear layer and reduce to 256 
    # nn.ReLU(), # introduce non linearity on the model 
    # nn.Linear(256, 7) #adding a linear layer and reduce to 256 
    # )

    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized SGD 
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9) #parameter lr = learnrate 

    # Optimizer adams 
    # optimizer_ft = Adam(model_ft.parameters(), lr=0.001)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) #reducteur de facteur de LR kann anpassen sein 
                                                                                 #gamma skalirer faktor
    
    ######################################################################
    # Train 
    # ^^^^^
    #
    
    train_name = 'model_resnet50_25epochs_dx7_imageRichtigVerteiltBlend'
    
    model_ft,train_losses,train_accs,val_losses,val_accs = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=25)

    ######################################################################
    # Save model
    # ^^^^^^^^^^
    #
    
    save_model(model_ft,train_name)

    ######################################################################
    # Confusion matrix
    # ^^^^^^^^^^^^^^^^
    #
    
    cf_name = train_name
    
    confusion_matrix_generate_test(model_ft,data_dir,cf_name)
    confusion_matrix_generate_train(model_ft,data_dir,cf_name)
    confusion_matrix_generate_val(model_ft,data_dir,cf_name)

    ######################################################################
    # Generate the accuracy curve
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #
    
    accuracy_curve_name = train_name
    
    accuracy_curve(train_losses,train_accs,val_losses,val_accs,accuracy_curve_name)

    ######################################################################
    # visualize_model 
    # ^^^^^^^^^^^^^^^
    #
    
    visualize_model(model_ft)
    plt.ioff()
    plt.show()

