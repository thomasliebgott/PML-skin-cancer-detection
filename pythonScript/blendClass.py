import os
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from PIL import Image

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

if __name__ == "__main__":
    
    countAKIEDC, countBCC, countBKL, countDF, countMEL, countNV, countVASC = counterFile()
    print('File count - AKIEDC : ', countAKIEDC)
    print('File count - BCC : ', countBCC)
    print('File count - BKL : ', countBKL)
    print('File count - DF : ', countDF)
    print('File count - MEL : ', countMEL)
    print('File count - NV : ', countNV)

    # Define a list of the functions
    functions = [miror, erosion, dilatation, rotation90, rotation180, rotation270, brightened75, brightened25]

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

                image = cv2.imread(filepath)
                filename = os.path.basename(filepath)
                cv2.imwrite(os.path.join(output_dir, filename), image)
                # Apply each function to the image and save the resulting images
                
                for func in functions:
                    
                    output_image = func(filepath)
                    
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
        else: 
            input_dir = r"D:\Hochschule\SS\PML\Project_PML\dx2\DF"
            output_dir = r"D:\Hochschule\SS\PML\Project_PML\dx2\DF"
                    
            for root, _, files in os.walk(input_dir):
                for file in files:
                    if file.endswith('.jpg'):
                        filepath = os.path.join(root, file)

                        image = cv2.imread(filepath)
                        filename = os.path.basename(filepath)
                        cv2.imwrite(os.path.join(output_dir, filename), image)
                        # Apply each function to the image and save the resulting images
                                
                        for func in functions:
                                    
                            output_image = func(filepath)
                                    
                            output_filename = os.path.splitext(file)[0] + "_" + func.__name__ + os.path.splitext(file)[1]
                                    
                            cv2.imwrite(os.path.join(output_dir, output_filename), output_image)

                            # Increment the image count
                            image_count += 1

                            # Break out of the loop if we have generated enough images
                            if image_count >= num_images:
                                    break    
                                    
                if image_count >= num_images:
                    break                    
