# -*- coding: utf-8 -*-
"""
Transfer Learning for Computer Vision Tutorial
==============================================
**Author**: `Sasank Chilamkurthy <https://chsasank.github.io>`_

In this tutorial, you will learn how to train a convolutional neural network for
image classification using transfer learning. You can read more about the transfer
learning at `cs231n notes <https://cs231n.github.io/transfer-learning/>`__

Quoting these notes,

    In practice, very few people train an entire Convolutional Network
    from scratch (with random initialization), because it is relatively
    rare to have a dataset of sufficient size. Instead, it is common to
    pretrain a ConvNet on a very large dataset (e.g. ImageNet, which
    contains 1.2 million images with 1000 categories), and then use the
    ConvNet either as an initialization or a fixed feature extractor for
    the task of interest.

These two major transfer learning scenarios look as follows:

-  **Finetuning the convnet**: Instead of random initialization, we
   initialize the network with a pretrained network, like the one that is
   trained on imagenet 1000 dataset. Rest of the training looks as
   usual.
-  **ConvNet as fixed feature extractor**: Here, we will freeze the weights
   for all of the network except that of the final fully connected
   layer. This last fully connected layer is replaced with a new one
   with random weights and only this layer is trained.

"""
# License: BSD
# Author: Sasank Chilamkurthy

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

cudnn.benchmark = True
plt.ion()   # interactive mode

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.

# Data augmentation and normalization for training
# Just normalization for validation

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
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict()) #copy le model 
    best_acc = 0.0
    
    # variables to generate the accuracy curve 
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    #error_values = []

    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                print('Training...')
            else:
                model.eval()   # Set model to evaluate mode
                print('Validating...')

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase.capitalize()} iteration', leave=False):
                inputs = inputs.to(device) #device est gpu 
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
# save the train model 
# ^^^^^^^^^^^^^^^^^^^^
#

def save_model(model_ft,train_name):
    model_folder = r'output\model'
    
    # Verify if the model folder already exist or not 
    folder_is_exists = True
    index_folder = 0
    while folder_is_exists:
        name_folder = f'model_{train_name}'
        if index_folder > 0:
            name_folder += f'_{index_folder}'
        model_path = os.path.join(model_folder, name_folder)
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
            folder_is_exists = False
        else:
            #add +1 if the folder already exist 
            index_folder += 1
    
    #save the model in the correct file 
    model_file_name = os.path.join(model_path, 'model.pth')
    print("model save")
    torch.save(model_ft.state_dict(), model_file_name)

######################################################################
# save the confusion matrix 
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#

def confusion_matrix_generate(model_ft,data_dir,cf_name):
    # load images 
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
    
    # calculate the confusion matrix 
    cf_matrix = confusion_matrix(true_labels, predicted_labels)
    
    # normalize the confusion matrix to 
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    
    #figure size 
    plt.figure(figsize=(12, 7))

    sn.heatmap(df_cm, annot=True)

    cm_folder = r'output\conf_matrix'

    # Verify if the confusionmaxtrix folder already exist or not 
    folder_is_exists = True
    index_folder = 0
    while folder_is_exists:
        name_folder = f'model_{cf_name}'
        if index_folder > 0:
            name_folder += f'_{index_folder}'
        cm_path = os.path.join(cm_folder, name_folder)
        if not os.path.exists(cm_path):
            os.makedirs(cm_path, exist_ok=True)
            folder_is_exists = False
        else:
            #add +1 if the folder already exist 
            index_folder += 1 
    #save the output 
    plt.savefig(os.path.join(cm_path, 'output.png'))  
    print("conf metrics save")
    plt.show()


######################################################################
# generate accuracy curve  
# ^^^^^^^^^^^^^^^^^^^^^^^
#

def accuracy_curve(train_losses,train_accs,val_losses,val_accs,accuracy_curve_name):
    output_folder = r'Project_PML\output\accuracy_curve'
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
    
######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrained model and reset final fully connected layer.
#

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
    # setup the train data
    # --------------------
    #
    
    data_dir = r'dx4'
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #sure graphic card 
    print(device)

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])

    model_ft = models.resnet34(pretrained=True) #nehmt das model
    num_ftrs = model_ft.fc.in_features #in_feature eingang auf unsere schicht 

    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    
    model_ft.fc = nn.Linear(num_ftrs, 7) #type de übetragungfuncktion #######################anderung 
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9) #parameter lr = learnrate 

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) #reducteur de facteur de LR kann anpassen sein 
                                                                                 #gamma skalirer faktor

    ######################################################################
    # Train 
    # ^^^^^
    #
    
    train_name = 'resnet34_25epochs_dx4'
    
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
    
    confusion_matrix_generate(model_ft,data_dir,cf_name)

    ######################################################################
    # generate the accuracy curve
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


    ######################################################################
    # ConvNet as fixed feature extractor
    # ----------------------------------
    #
    # Here, we need to freeze all the network except the final layer. We need
    # to set ``requires_grad = False`` to freeze the parameters so that the
    # gradients are not computed in ``backward()``.
    #
    # You can read more about this in the documentation
    # `here <https://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward>`__.
    #

    #model_conv = torchvision.models.resnet18(pretrained=True)
    #for param in model_conv.parameters():
    #    param.requires_grad = False
#
    ## Parameters of newly constructed modules have requires_grad=True by default
    #num_ftrs = model_conv.fc.in_features
    #model_conv.fc = nn.Linear(num_ftrs, 2)
#
    #model_conv = model_conv.to(device)
#
    #criterion = nn.CrossEntropyLoss()
#
    ## Observe that only parameters of final layer are being optimized as
    ## opposed to before.
    #optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
#
    ## Decay LR by a factor of 0.1 every 7 epochs
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
#
#
    #######################################################################
    ## Train and evaluate
    ## ^^^^^^^^^^^^^^^^^^
    ##
    ## On CPU this will take about half the time compared to previous scenario.
    ## This is expected as gradients don't need to be computed for most of the
    ## network. However, forward does need to be computed.
    ##
#
    #model_conv = train_model(model_conv, criterion, optimizer_conv,
    #                        exp_lr_scheduler, num_epochs=25)
#
    #######################################################################
    ##
#
    #visualize_model(model_conv)
#
    #plt.ioff()
    #plt.show()

    ######################################################################
    # Further Learning
    # -----------------
    #
    # If you would like to learn more about the applications of transfer learning,
    # checkout our `Quantized Transfer Learning for Computer Vision Tutorial <https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html>`_.
    #
    #