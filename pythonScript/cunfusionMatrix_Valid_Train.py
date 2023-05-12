
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #sure graphic card 

def confusionMatrix(model, testloader):
    y_pred = []
    y_true = []

    # iterate over test data

    for  inputs, labels in testloader['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs, labels = inputs.cuda(), labels.cuda()
            
            output = model(inputs) # Feed Network
            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction
            
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth

    # constant for classes
    classes = ('AKIEDC', 'BCC', 'BKL', 'MEL', 'NV', 'VASC')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('output.png')


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

if __name__ == '__main__':
# Data augmentation and normalization for training
# Just normalization for validation
    ######################################################################
    # Visualizing the model predictions
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #
    # Generic function to display predictions for a few images
    

    data_transforms = {
    'test': transforms.Compose([
        transforms.Resize((450, 600)),
        transforms.CenterCrop(450),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }


    data_dir = 'D:\\pml\\branch_Ali\\PML-master\\PML-master\\inputData3'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['test']}

    model_ft = models.resnet18(pretrained=True) #nehmt das model
    num_ftrs = model_ft.fc.in_features #in_feature eingang auf unsere schicht 

    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 7) #type de übetragungfuncktion #######################anderung 

    model_ft.load_state_dict(torch.load("D:\\pml\\ali_branch\\PML\\model\\model_2\\model.pth"))
    model_ft.eval()

    confusionMatrix(model_ft, dataloaders)




