import os
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from PIL import Image
import shutil 

def counterFile(path):
    
    countAKIEDC = 0
    # Join path with 'AKIEDC' directory and iterate over files
    for filename in os.listdir(os.path.join(path, 'AKIEDC')):
        # Check if current path is a file
        if os.path.isfile(os.path.join(path, 'AKIEDC', filename)):
            countAKIEDC += 1
    
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

    # load image with the directory
    img = cv2.imread(directory)
    
    # aplly the erosion 
    dist = cv2.erode(img, element)
    
    return dist

def dilatation(directory):
    dilation_size = 1 
    dilation_shape = cv2.MORPH_ELLIPSE
    
    element = cv2.getStructuringElement(dilation_shape, (2 * dilation_size + 1, 2 * dilation_size + 1), (dilation_size, dilation_size))

    # Charger l'image à partir du chemin de répertoire
    img = cv2.imread(directory)
    
    # Appliquer l'opération de dilatation sur l'image chargée
    dist = cv2.dilate(img, element)
    
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

    # Convert PIL Image to OpenCV image
    opencv_img = cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2BGR)
    
    return opencv_img

def rotation180(directory):
    # Open the original image
    img = cv2.imread(directory)
    
    # Rotate the image 180 degrees
    rotated_img = cv2.rotate(img, cv2.ROTATE_180)

    return rotated_img

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

    # Convert PIL Image to OpenCV image
    opencv_img = cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2BGR)
    
    return opencv_img

def brightened75(directory):
    # Open the original image
    img = Image.open(directory)

    # Apply brightness enhancement
    brightened_img = ImageEnhance.Brightness(img).enhance(0.75)

    # Convert PIL Image to NumPy array and then to OpenCV image
    opencv_img = cv2.cvtColor(np.array(brightened_img), cv2.COLOR_RGB2BGR)

    return opencv_img

def brightened25(directory):
    # Open the original image
    img = Image.open(directory)

    # Apply brightness enhancement
    brightened_img = ImageEnhance.Brightness(img).enhance(1.25)

    # Convert PIL Image to NumPy array and then to OpenCV image
    opencv_img = cv2.cvtColor(np.array(brightened_img), cv2.COLOR_RGB2BGR)

    return opencv_img   

def applyImageProcessing(input_dir,output_dir,num_images,functions):
    # Loop over the input images and apply the functions to generate new images
    image_count = 0

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_count = 0
                input_filepath = os.path.join(root, file)
                output_filepath = os.path.join(output_dir, os.path.relpath(input_filepath, input_dir))

                # Create the output directory if it does not exist
                output_directory = os.path.dirname(output_filepath)
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)

                # Copy the input image to the output directory
                shutil.copy(input_filepath, output_filepath)

                # Apply each function to the image and save the resulting images
                for func in functions:
                    output_image = func(input_filepath)
                    output_filename = os.path.splitext(file)[0] + "_" + func.__name__ + os.path.splitext(file)[1]
                    output_filepath = os.path.join(output_dir, os.path.relpath(input_filepath, input_dir))
                    output_filepath = os.path.join(output_directory, output_filename)
                    cv2.imwrite(output_filepath, output_image)

                    # Increment the image count
                    image_count += 1

                    # Break out of the loop if we have generated enough images
                    if image_count >= num_images:
                        break
                    
                    # Break out of the loop if we have generated enough images
        if image_count >= num_images:
            break

if __name__ == "__main__":
    
    # Define the input and output directories
    input_dir = r"D:\Hochschule\SS\PML\Project_PML\dx"
    output_dir = r"D:\Hochschule\SS\PML\Project_PML\dx2"
    
    
    countAKIEDC, countBCC, countBKL, countDF, countMEL, countNV, countVASC = counterFile(input_dir)
    print('File count - AKIEDC : ', countAKIEDC)
    print('File count - BCC : ', countBCC)
    print('File count - BKL : ', countBKL)
    print('File count - DF : ', countDF)
    print('File count - MEL : ', countMEL)
    print('File count - NV : ', countNV)

    # Define a list of the functions
    functions = [miror, erosion, dilatation, rotation90, rotation180, rotation270, brightened75, brightened25]

    # Define the number of images to generate
    num_images = 3500

    # Loop over the input images and apply the functions to generate new images
    image_count = 0
    
    applyImageProcessing(input_dir,output_dir,num_images,functions)
    
    countAKIEDC, countBCC, countBKL, countDF, countMEL, countNV, countVASC = counterFile(output_dir)
    print('File count - AKIEDC : ', countAKIEDC)
    print('File count - BCC : ', countBCC)
    print('File count - BKL : ', countBKL)
    print('File count - DF : ', countDF)
    print('File count - MEL : ', countMEL)
    print('File count - NV : ', countNV)
    


