import cv2
import numpy as np
from PIL import Image
import os 
import re 

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

    return resized_img 

    # # Extract the image ID from the path using regular expressions
    # image_id = re.search(r'ISIC_\d+', directory).group()

    # # Save the resized image in the same file format as the original image
    # resized_img.save(os.path.join("D:\Hochschule\SS\PML\Project_PML/rotation", image_id+"_rotated_270.jpg"))

directory = "dx/AKIEDC/ISIC_0024329.jpg"
resized_img = rotation90(directory)

image_id = re.search(r'ISIC_\d+', directory).group() 
resized_img.save(os.path.join("D:\Hochschule\SS\PML\Project_PML/rotation", image_id+"_rotated_90.jpg"))

resized_img = rotation180(directory)

image_id = re.search(r'ISIC_\d+', directory).group() 
resized_img.save(os.path.join("D:\Hochschule\SS\PML\Project_PML/rotation", image_id+"_rotated_180.jpg"))

resized_img = rotation270(directory)
image_id = re.search(r'ISIC_\d+', directory).group() 
resized_img.save(os.path.join("D:\Hochschule\SS\PML\Project_PML/rotation", image_id+"_rotated_270.jpg"))


