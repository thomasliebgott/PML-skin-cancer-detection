import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 7)

file_name = "model_resnet18_10epochs_dx4"

model_ft.load_state_dict(torch.load(r"C:\Users\Thomaslieb\Desktop\PML\output\model\model_resnet18_10epochs_dx4\model.pth"))
model_ft = model_ft.to(device)
model_ft.eval()

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize((450, 600)),
        transforms.CenterCrop(450),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = r'dx4'

image_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])

dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=4, shuffle=True, num_workers=4)

true_labels = []
predicted_labels = []

if __name__ == '__main__':
    model_ft.eval()  
    for inputs, labels in dataloader:
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

    # caclul the different evaluations metrics
    precision = precision_score(true_labels, predicted_labels, average=None)
    recall = recall_score(true_labels, predicted_labels, average=None)
    f1score = f1_score(true_labels, predicted_labels, average=None)
    accuracy = np.sum(true_labels == predicted_labels) / len(true_labels)
    specificity = recall_score(true_labels, predicted_labels, average='weighted')

    # Round the metrics 
    precision = np.round(precision, 4)
    recall = np.round(recall, 4)
    f1 = np.round(f1score, 4)
    accuracy = np.round(accuracy, 4)
    specificity = np.round(specificity, 4)

    # create dictionnary to store the differents values 
    data_dictio = {'classes': classes, 'Precision': precision, 'Recall': recall, 'F1-score': f1score}
    dataFrame = pd.DataFrame(data_dictio)

    evaluation_metrics_data = {'classes': ['Accuracy', 'Specificity'],
                    'Precision': [accuracy, specificity],
                    'Recall': ['', ''],
                    'F1_score': ['', '']}

    # Creattion of the coluum
    dataFrame = pd.concat([dataFrame, pd.DataFrame(evaluation_metrics_data)], ignore_index=True)

    # adapt value '' for csv file 
    dataFrame.replace('', np.nan, inplace=True)
    
    # convert values into float
    num_values = ['Precision', 'Recall', 'F1-score']
    dataFrame[num_values] = dataFrame[num_values].astype(float)

    # Round the metrics 
    dataFrame = dataFrame.round({'Precision': 4, 'Recall': 4, 'F1-score': 4})
    
    #save the csv. file
    output_dir = r'output\metrics'
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'{file_name}.csv')
    dataFrame.to_csv(file_path, index=False)