import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import shutil 
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

model_ft = models.resnet50(pretrained=True) 
num_ftrs = model_ft.fc.in_features 

model_ft.fc = nn.Linear(num_ftrs, 7) 

model_dir = r'output\model\model_resnet50_25epochs_dx7-imageRichtigVerteiltBlend_Linear\model.pth'

model_ft.load_state_dict(torch.load(model_dir))
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

data_dir = r'dx7_imageRichtigVerteiltBlend'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                        data_transforms[x])
                for x in ['test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                            shuffle=False, num_workers=4)
            for x in ['test']}

true_labels = []
predicted_labels = []

if __name__ == '__main__':
    model_ft.eval()  #start to put the model in evalutation mode 
    #load the dataloaders to get the test values and save the predicted value and true value in the tab
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
    
    # output dir for the images 
    output_dir = r'comparisonPredictedError'

    # look at the true_class and predicted_class
    for pred in tqdm(range(len(predicted_labels)), desc='Saving Images', unit='image', leave=False):
        predicted_label = predicted_labels[pred]
        true_label = true_labels[pred]
        
        # get info of the true_class and predicted_class
        predicted_class = classes[predicted_label]
        true_class = classes[true_label]
        
        # get path of the images that is tested
        image_path = image_datasets['test'].imgs[pred][0]
        image_name = os.path.basename(image_path)
        
        if predicted_label != true_label:
            subfolder_dir = os.path.join(output_dir, true_class, predicted_class)
        else:
            subfolder_dir = os.path.join(output_dir, true_class, true_class)
        
        # create dir if it doesn't exist
        if not os.path.exists(subfolder_dir):
            os.makedirs(subfolder_dir, exist_ok=True)
        
        destination_dir = os.path.join(subfolder_dir, image_name)
        shutil.copy(image_path, destination_dir)

    

