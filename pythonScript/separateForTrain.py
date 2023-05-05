import os
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from PIL import Image
import shutil 

def getMax(countAKIEDC,countBCC,countBKL,countDF,countMEL,countNV,countVASC):
    nn = [countAKIEDC,countBCC,countBKL,countDF,countMEL,countNV,countVASC]
    max = np.max(nn)
    return max 

def counterFile(path):
    
    countAKIEDC = 0
    # Join path with 'AKIEDC' directory and iterate over files
    for filename in os.listdir(os.path.join(path, 'AKIEDC')):
        # Check if current path is a file
        if os.path.isfile(os.path.join(path, 'AKIEDC', filename)):
            countAKIEDC += 1
    
    countBCC = 0
    for filename in os.listdir(os.path.join(path,'BCC')):
        # check if current path is a file
        if os.path.isfile(os.path.join(path,'BCC', filename)):
            countBCC += 1

    countBKL = 0
    for filename in os.listdir(os.path.join(path,'BKL')):
        # check if current path is a file
        if os.path.isfile(os.path.join(path,'BKL', filename)):
            countBKL += 1
    #print('File count - BKL : ', countBCC)

    countDF = 0
    for filename in os.listdir(os.path.join(path,'DF')):
        # check if current path is a file
        if os.path.isfile(os.path.join(path,'DF', filename)):
            countDF += 1
    #print('File count - DF : ', countDF)

    countMEL = 0
    for filename in os.listdir(os.path.join(path,'MEL')):
        # check if current path is a file
        if os.path.isfile(os.path.join(path,'MEL', filename)):
            countMEL += 1
    #print('File count - MEL : ', countMEL)

    countNV = 0
    for filename in os.listdir(os.path.join(path,'NV')):
        # check if current path is a file
        if os.path.isfile(os.path.join(path,'NV', filename)):
            countNV += 1
    #print('File count - NV : ', countNV)

    countVASC = 0
    for filename in os.listdir(os.path.join(path,'VASC')):
        # check if current path is a file
        if os.path.isfile(os.path.join(path,'VASC', filename)):
            countVASC += 1
    #print('File count - VASC : ', countVASC)
    
    return countAKIEDC, countBCC, countBKL, countDF, countMEL, countNV, countVASC

def separtationTrainVal(input_dir,output_dir,trainPourcentage):
    
        train_dir = os.path.join(output_dir, "train")
        val_dir = os.path.join(output_dir, "val")
        
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)   
        
        if not os.path.exists(val_dir):
            os.makedirs(val_dir) 

        countAKIEDC, countBCC, countBKL, countDF, countMEL, countNV, countVASC = counterFile(input_dir)
        num_images = getMax(countAKIEDC, countBCC, countBKL, countDF, countMEL, countNV, countVASC)
        
        train_image_number = round(trainPourcentage * num_images / 100)
        print("nombre image in train = ", train_image_number)
        val_image_number = train_image_number - num_images
            
        for originalRoot, directorys, _ in os.walk(input_dir):
            for directory in directorys:
                input_dir_path = os.path.join(originalRoot, directory)
                output_dir_path = os.path.join(output_dir, directory)
        
        # Create the output directory if it does not exist
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
                
                

if __name__ == "__main__": 
    
    # Define the input and output directories
    input_dir = r"D:\Hochschule\SS\PML\Project_PML\dx2"
    output_dir = r"D:\Hochschule\SS\PML\Project_PML\dx3" 
    
    
    