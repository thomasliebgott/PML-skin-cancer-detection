import os
from PIL import Image, ImageEnhance
import shutil
import cv2
import numpy as np
from PIL import Image
import re 
import glob


directory = "D:\Hochschule\SS\PML\Project_PML\dx"

def counterFile():

    countAKIEDC = 0
    # Iterate directory
    for path in os.listdir('dx/AKIEDC'):
        # check if current path is a file
        if os.path.isfile(os.path.join('dx/AKIEDC', path)):
            countAKIEDC += 1
    #print('File count - AKIEDC : ', countAKIEDC)
    
    countBCC = 0
    for path in os.listdir('dx/BCC'):
        # check if current path is a file
        if os.path.isfile(os.path.join('dx/BCC', path)):
            countBCC += 1
    #print('File count - BCC : ', countBCC)
    
    countBKL = 0
    for path in os.listdir('dx/BKL'):
        # check if current path is a file
        if os.path.isfile(os.path.join('dx/BKL', path)):
            countBKL += 1
    #print('File count - BKL : ', countBCC)   
      
    countDF = 0      
    for path in os.listdir('dx/DF'):
        # check if current path is a file
        if os.path.isfile(os.path.join('dx/DF', path)):
            countDF += 1
    #print('File count - DF : ', countDF)
        
    countMEL = 0
    for path in os.listdir('dx/MEL'):
        # check if current path is a file
        if os.path.isfile(os.path.join('dx/MEL', path)):
            countMEL += 1
    #print('File count - MEL : ', countMEL)

    countNV = 0
    for path in os.listdir('dx/NV'):
        # check if current path is a file
        if os.path.isfile(os.path.join('dx/NV', path)):
            countNV += 1
    #print('File count - NV : ', countNV)
        
    countVASC = 0   
    for path in os.listdir('dx/VASC'):
        # check if current path is a file
        if os.path.isfile(os.path.join('dx/VASC', path)):
            countVASC += 1
    #print('File count - VASC : ', countVASC)
    
    return countAKIEDC, countBCC, countBKL, countDF, countMEL, countNV, countVASC

def miror(directory):
    # Load the image
    img = cv2.imread(directory)

    # Flip the image horizontally
    img_flip = cv2.flip(img, 1)

    # Save the flipped image
    return img_flip

def erosion(directory):
    erosion_size = 1 
    erosion_shape = cv2.MORPH_ELLIPSE
    
    element = cv2.getStructuringElement(erosion_shape ,(2 * erosion_size +1, 2* erosion_size +1),(erosion_size, erosion_size))
    
    dist = cv2.erode(directory, element)
    return dist 

def dilatation(directory):
    dilatation_size = 1 
    dilatation_shape = cv2.MORPH_ELLIPSE

    element = cv2.getStructuringElement(dilatation_shape,(2 * dilatation_size +1, 2* dilatation_size +1),(dilatation_size, dilatation_size))
    
    dist = cv2.dilate(directory, element)
    return dist 

def rotation90(directory):
    # Open the original image
    img = Image.open(directory)
    
    # Rotate the image 90 degrees clockwise
    rotated_img = img.rotate(-90, expand=True)

    # Determine the aspect ratio of the original image
    width, height = img.size
    aspect_ratio = width / height

    # Resize the rotated image to the original dimensions
    resized_img = rotated_img.resize((int(height * aspect_ratio), height))

    return resized_img

    # # Extract the image ID from the path using regular expressions
    # image_id = re.search(r'ISIC_\d+', directory).group()
    
    # # Save the resized image in the same file format as the original image
    # resized_img.save(os.path.join("D:\Hochschule\SS\PML\Project_PML/rotation", image_id+"_rotated_90.jpg"))

def rotation180(directory):
    # Open the original image
    img = Image.open(directory)

    # Rotate the image 180 degrees clockwise
    rotated_img = img.rotate(180, expand=True)

    return rotated_img
    # # Save the rotated image
    # # Extract the image ID from the path using regular expressions
    # image_id = re.search(r'ISIC_\d+', directory).group()
    # rotated_img.save(os.path.join("D:\Hochschule\SS\PML\Project_PML/rotation", image_id+"_rotated_180.jpg"))

def rotation270(directory):
    # Open the original image
    img = Image.open(directory)

    # Rotate the image 270 degrees clockwise
    rotated_img = img.rotate(90, expand=True)

    # Determine the aspect ratio of the original image
    width, height = img.size
    aspect_ratio = width / height

    # Resize the rotated image to the original dimensions
    resized_img = rotated_img.resize((int(height * aspect_ratio), height))

    # Extract the image ID from the path using regular expressions
    image_id = re.search(r'ISIC_\d+', directory).group()

    # Save the resized image in the same file format as the original image
    resized_img.save(os.path.join("D:\Hochschule\SS\PML\Project_PML/rotation", image_id+"_rotated_270.jpg"))

def brightened75(directory):
        # Check if the filename matches the required pattern
        image = Image.open(directory)
        brightened_image = ImageEnhance.Brightness(image).enhance(0.75)
        return brightened_image   
 
def brightened25(directory):
        # Check if the filename matches the required pattern
        image = Image.open(directory)
        brightened_image = ImageEnhance.Brightness(image).enhance(0.25)
        return brightened_image      
               
if __name__ == "__main__":
    
    countAKIEDC, countBCC, countBKL, countDF, countMEL, countNV, countVASC = counterFile()
    print('File count - AKIEDC : ', countAKIEDC)
    print('File count - BCC : ', countBCC)
    print('File count - BKL : ', countBKL)
    print('File count - DF : ', countDF)
    print('File count - MEL : ', countMEL)
    print('File count - NV : ', countNV)

    # Define a list of the functions
    functions = [erosion, dilatation, miror, rotation90, rotation180, rotation270, brightened75, brightened25]

    # Define the input and output directories
    input_dir = r"D:\Hochschule\SS\PML\Project_PML\dx\DF"
    output_dir = r"D:\Hochschule\SS\PML\Project_PML\dx2\DF"

    # Define the number of images to generate
    num_images = 3500

    # Define the list of image transformation functions to apply
    functions = [miror, erosion, dilatation, rotation90, rotation180, rotation270, brightened75, brightened25]

    # Loop over the input images and apply the functions to generate new images
    image_count = 0
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.jpg'):
                filepath = os.path.join(root, file)

                # Load the image
                image = cv2.imread(filepath)

                # Apply each function to the image and save the resulting images
                for func in functions:
                    output_image = func(image)
                    
                    output_filename = os.path.splitext(file)[0] + "_" + func.__name__ + os.path.splitext(file)[1]
                    
                    cv2.imwrite(os.path.join(output_dir, output_filename), output_image)

                    # Increment the image count
                    image_count += 1

                    # Break out of the loop if we have generated enough images
                    if image_count >= num_images:
                        break

                # Break out of the loop if we have generated enough images
                if image_count >= num_images:
                    break

        # Break out of the loop if we have generated enough images
        if image_count >= num_images:
            break

