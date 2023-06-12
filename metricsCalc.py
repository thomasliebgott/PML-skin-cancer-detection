import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import numpy as np
from sklearn.metrics import recall_score, f1_score, precision_score
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load the model type
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 7) # chnage if it's another last layer type 

fileNameModel = "model_resnet18_1epochs_dx4_ohneHaareEntfernung"

# load the model 
model_ft.load_state_dict(torch.load(r"D:\PML\Project_PML\output\model\model_resnet18_1epochs_dx4_ohneHaareEntfernung\model.pth"))
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

data_dir = r'dx4_ohneHaareEntfernung'

image_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])

dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=4, shuffle=True, num_workers=4)

true_labels = []
predicted_labels = []


if __name__ == '__main__':
    model_ft.eval()  # load the model and setup in evaluate mode
    
    # collect the true/false prediction 
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)
        predicted_labels.extend(preds)  

        labels = labels.data.cpu().numpy()
        true_labels.extend(labels) 

    classes = ('AKIEDC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC')

    #convert prediction in tensor 
    predicted_labels = torch.tensor(predicted_labels)
    true_labels = torch.tensor(true_labels)

    #convert tensor to numpy 
    predicted_labels = predicted_labels.cpu().numpy()
    true_labels = true_labels.cpu().numpy()

    #calcul of the different evaluation metrics 
    
    recall = recall_score(true_labels, predicted_labels, average=None)
    f1score = f1_score(true_labels, predicted_labels, average=None)
    precision = precision_score(true_labels, predicted_labels, average=None)
    accuracy = np.sum(true_labels == predicted_labels) / len(true_labels)
    specificity = recall_score(true_labels, predicted_labels, average='weighted')

    #round the values 
    recall = np.round(recall, 2)
    f1score = np.round(f1score, 2)
    precision = np.round(precision, 2)
    accuracy = np.round(accuracy, 2)
    specificity = np.round(specificity, 2)

    dataValues = {'classes': classes, 'Recall': recall, 'F1-score': f1score, 'Precision': precision}
    dataFrame = pd.DataFrame(dataValues)

    evaluation_metrics_data = {'classes': ['Accuracy', 'Specificity'],
                               'Recall': [accuracy, specificity],
                               'F1_score': ['', ''],
                               'Precision': ['', '']}

    dataFrame = pd.concat([dataFrame, pd.DataFrame(evaluation_metrics_data)], ignore_index=True) #merge the data into one ligne
    dataFrame.replace('', np.nan, inplace=True) #replace values of '' by NaN
    num_values = ['Recall', 'F1-score', 'Precision'] #create liste with collums of the dataFrame
    dataFrame[num_values] = dataFrame[num_values].astype(float) #convert num_values into a float
    dataFrame = dataFrame.round({'Recall': 2, 'F1-score': 2, 'Precision': 2}) #round the values 2nd decimal
    
    output_dir = r'output/metrics'
    
    os.makedirs(output_dir, exist_ok=True)
    
    csv_file_path = os.path.join(output_dir, f'{fileNameModel}.csv') #create name of the file 
    dataFrame.to_csv(csv_file_path, index=False) #save the data of the dataFrame into the csv 
    df = pd.read_csv(csv_file_path)

    # get the values of the df
    classes = df['classes']
    f1_scores = df['F1-score']
    recalls = df['Recall']

    colors = ['blue', 'orange']

    # setup the positions of the different classes on the plot regulary spaces
    step = np.arange(len(classes))

    chart_size = 0.35

    # setup the dimension of the figure
    figure, graph_axe = plt.subplots(figsize=(12, 6))

    # trace the charts f1_score
    graph_axe.bar(step, f1_scores, width=chart_size, color=colors[0], label='F1-score')

    # trace the charts recall
    graph_axe.bar(step + chart_size, recalls, width=chart_size, color=colors[1], label='Recall')

    # add the name of the classes 
    graph_axe.set_xticks(step + chart_size / 2)
    graph_axe.set_xticklabels(classes)

    # Setup graph
    graph_axe.legend()
    graph_axe.set_xlabel('Classes')
    graph_axe.set_ylabel('Scores')
    graph_axe.set_title('Result evaluations metrics for : ' + fileNameModel)

    for i in range(len(classes)):
        val_f1_score = round(max(f1_scores), 2)
        val_recall = round(max(recalls), 2)
        #adding the value of the score on every chart
        graph_axe.annotate(f'{val_f1_score}', (step[i], f1_scores[i]), xytext=(0, 2.5),
                    textcoords='offset points', ha='center', color='black')
        graph_axe.annotate(f'{val_recall}', (step[i] + chart_size, recalls[i]), xytext=(0, 2.5),
                    textcoords='offset points', ha='center', color='black')

    # save the file
    output_file_path = os.path.join(output_dir, f'{fileNameModel}.png')
    plt.savefig(output_file_path, dpi=300)

    # show graph
    # plt.show()