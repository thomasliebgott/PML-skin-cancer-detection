##This function separate the images before the data augmentation V3 

import os
import random
import shutil

# input folder with images already separated
input_folder = r'dx_OriginalVerteiltImages'

# path for training und validation
output_train_folder = r'dx6_imageRichtigVerteilt/train'
output_val_folder = r'dx6_imageRichtigVerteilt/val'

# train images pourcentage
train_percentage = 0.8

# Parcourir chaque sous-dossier dans le dossier principal
for directoryPath, directoryNames, files in os.walk(input_folder):
    for dir_name in directoryNames:
        dir_path = os.path.join(directoryPath, dir_name)
        
        # Creation of a list with all image from subfolder 
        imagesName =[]
        
        # go on file
        for file_name in os.listdir(dir_path):
            # get file path
            file_path = os.path.join(dir_path, file_name)
            
            if os.path.isfile(file_path): #if the file exist 
                #append the name to the list of images
                imagesName.append(file_name)
                
        
        # calcultate the number of images for val and train
        num_train = int(len(imagesName) * train_percentage)
        num_val = len(imagesName) - num_train
        
        # shuffle the images to get random
        random.shuffle(imagesName)
        
        # divide the images in 2 list with number of train and val
        train_images = imagesName[:num_train]
        val_images = imagesName[num_train:]
        
        # Creation of the folder for each caterogie
        output_train_dir = os.path.join(output_train_folder, dir_name) #dire_name take the name of the orginial 
        output_val_dir = os.path.join(output_val_folder, dir_name)
        os.makedirs(output_train_dir, exist_ok=True)
        os.makedirs(output_val_dir, exist_ok=True)
        
        # move the images
        for i in train_images:
            src = os.path.join(dir_path, i)
            dst = os.path.join(output_train_dir, i)
            shutil.copy(src, dst)
        
        # Déplacer les images de validation vers le dossier de sortie correspondant
        for i in val_images:
            src = os.path.join(dir_path, i)
            dst = os.path.join(output_val_dir, i)
            shutil.copy(src, dst)

print("end ")
