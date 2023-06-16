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
import os

from sklearn.metrics import confusion_matrix

import seaborn as sn
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

def confusionMatrix(model, testloader):
    predicted_labels = []
    true_labels = []

    for inputs, labels in testloader['test']:
        inputs = inputs.to(device)  
        labels = labels.to(device) 

        output = model(inputs)  
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        predicted_labels.extend(output)  

        labels = labels.data.cpu().numpy()
        true_labels.extend(labels)  

    for inputs, labels in testloader['val']:
        inputs = inputs.to(device)  
        labels = labels.to(device)  

        output = model(inputs)  
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        predicted_labels.extend(output)  

        labels = labels.data.cpu().numpy()
        true_labels.extend(labels)  

    classes = ('AKIEDC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC')

    # Build confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    df = pd.DataFrame(cm / np.sum(cm, axis=1)[:, None], index=classes, columns=classes) 
    
    plt.figure(figsize=(12, 7))
    axes = sn.heatmap(df, annot=True)
    axes.set_title("val Confusion Matrix")
    axes.set_xlabel("Predicted labels")
    axes.set_ylabel("True labels")
    
    cm_folder_save = r'output\conf_matrix_val_test'
    
    plt.savefig(os.path.join(cm_folder_save,'model_resnet50_25epochs_dx7-imageRichtigVerteiltBlend_Linear.png'))

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

    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize((450, 600)),
            transforms.CenterCrop(450),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((450, 600)),
            transforms.CenterCrop(450),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    data_dir = 'D:\PML\Project_PML\dx7_imageRichtigVerteiltBlend'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['test', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                  shuffle=True, num_workers=4)
                   for x in ['test', 'val']}

    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, 7)

    model_ft.load_state_dict(
        torch.load("D:\PML\Project_PML\output\model\model_resnet50_25epochs_dx7-imageRichtigVerteiltBlend_Linear\model.pth",
                   map_location=device))  # Move the model to the same device
    model_ft = model_ft.to(device)
    model_ft.eval()

    confusionMatrix(model_ft, dataloaders)
