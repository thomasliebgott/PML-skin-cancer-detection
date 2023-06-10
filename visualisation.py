import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #sure graphic card 

model_ft = models.resnet18(pretrained=True) #nehmt das model
num_ftrs = model_ft.fc.in_features #in_feature eingang auf unsere schicht 

# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 7) #type de übetragungfuncktion #######################anderung 

model_ft.load_state_dict(torch.load(r'output\model\model_3\model.pth'))
model_ft = model_ft.to(device)
model_ft.eval()

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

data_dir = r'dx3'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                        data_transforms[x])
                for x in ['test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                            shuffle=True, num_workers=4)
            for x in ['test']}

true_labels = []
predicted_labels = []

def save_confusion_matrix(visualisation_name):
    
    cm_folder = r'output\conf_matrix'
    
    # Verify if the confusionmaxtrix folder already exist or not 
    folder_is_exists = True
    index_folder = 0
    while folder_is_exists:
        name_folder = f'model_{visualisation_name}'
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

# collect the true/false prediction 
if __name__ == '__main__':
    model_ft.eval()  
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
    
    cf_matrix = confusion_matrix(true_labels, predicted_labels)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)

    name_training = 'testlabel_and_true'
    
    save_confusion_matrix(name_training)
    

