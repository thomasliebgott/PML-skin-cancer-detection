import os
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from PIL import Image
import shutil 


def separtationTrainVal(input_dir,output_dir,trainPourcentage,valPourcentage):
    for originalRoot, directorys, _ in os.walk(input_dir):
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)
                
                
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
    
    