##This function separate the images after the data augmentation

import os
import random
import shutil

input_dir = r"dx2_Blend"
output_dir = r"dx4_ohneHaareEntfernung"

train_percentage = 80
val_percentage = 20
#test_percentage = 10

for currentDirectory, subDirectories, files in os.walk(input_dir): 
    for directory_name in subDirectories:
        
        #setup name of different directory
        input_subdir = os.path.join(currentDirectory, directory_name)
        
        output_train = os.path.join(output_dir, "train", directory_name)
        output_val = os.path.join(output_dir, "val", directory_name)
        #output_test = os.path.join(output_dir, "test", directoryName)
        
        #create the directory if theyr don't exist
        if not os.path.exists(output_train):
            os.makedirs(output_train)
            
        if not os.path.exists(output_val):
            os.makedirs(output_val)
            
        # if not os.path.exists(output_test):
        #     os.makedirs(output_test)
            
        # setup list with all the differents image in directory
        imageName = [f for f in os.listdir(input_subdir) if f.endswith(".jpg")]
        
        # setup random image order in the list
        random.shuffle(imageName)
        
        # calculate the number of images for training validation and test
        numImages = len(imageName)

        num_train_images = int(numImages * train_percentage / 100)
        num_val_images = int(numImages * val_percentage / 100)
        #num_test_images = int(num_images * test_percentage / 100)
        
        #train = imageName[:num_train_images]
        #val = imageName[num_train_images:num_train_images+num_val_images]
        #test = imageName[num_train_images+num_val_images:]
        
        # Copy the training images to the output directory
        train_imageName = imageName[:num_train_images]
        for filename in train_imageName:
            input_filepath = os.path.join(input_subdir, filename)
            output_filepath = os.path.join(output_train, filename)
            shutil.copy(input_filepath, output_filepath)
        
        # Copy the validation images to the output directory
        val_imageName = imageName[num_train_images:num_train_images+num_val_images]
        for filename in val_imageName:
            input_filepath = os.path.join(input_subdir, filename)
            output_filepath = os.path.join(output_val, filename)
            shutil.copy(input_filepath, output_filepath)
        
        # Copy the test images to the output directory    
        # test_imageName = imageName[num_train_images+num_val_images:]
        # for filename in test_imageName:
        #     input_filepath = os.path.join(input_subdir, filename)
        #     output_filepath = os.path.join(output_test, filename)
        #     shutil.copy(input_filepath, output_filepath)
