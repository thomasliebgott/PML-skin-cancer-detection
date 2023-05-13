from PIL import Image, ImageEnhance
import os

# Set the directory containing the images to process
# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "..\Project_PML"))

# Set the directory containing the images to process
directory = os.path.join(parent_dir, "rotation/")

idImage= "ISIC_0024329"

# Loop through each file in the directory
def brightened75(directory,idImage):
    for filename in os.listdir(directory):
        # Check if the filename matches the required pattern
        if filename.startswith(idImage) and filename.endswith("_rotated_90.jpg"):
            image = Image.open(directory + filename)
            brightened_image = ImageEnhance.Brightness(image).enhance(0.75)
            # Save the modified image with a new filename
            new_filename = filename.split(".")[0] + "_brightened_0.75.jpg" # add suffix "_brightened" before file extension
            brightened_image.save(directory + new_filename)
            
            # Repeat the same code for "_rotated_180" and "_rotated_270"
        if filename.startswith(idImage) and filename.endswith("_rotated_180.jpg"):
            image = Image.open(directory + filename)
            brightened_image = ImageEnhance.Brightness(image).enhance(0.75)
            # Save the modified image with a new filename
            new_filename = filename.split(".")[0] + "_brightened_0.75.jpg"  # add suffix "_brightened" before file extension
            brightened_image.save(directory + new_filename)

        if filename.startswith(idImage) and filename.endswith("_rotated_270.jpg"):
            image = Image.open(directory + filename)
            brightened_image = ImageEnhance.Brightness(image).enhance(1.25)
            # Save the modified image with a new filename
            new_filename = filename.split(".")[0] + "_brightened_0.75.jpg"  # add suffix "_brightened" before file extension
            brightened_image.save(directory + new_filename)
            
def brightened25(directory,idImage):
    for filename in os.listdir(directory):            
        if filename.startswith(idImage) and filename.endswith("_rotated_90.jpg"):
            image = Image.open(directory + filename)
            brightened_image = ImageEnhance.Brightness(image).enhance(1.25)
            # Save the modified image with a new filename
            new_filename = filename.split(".")[0] + "_brightened_1.25.jpg"  # add suffix "_brightened" before file extension
            brightened_image.save(directory + new_filename)

            # Repeat the same code for "_rotated_180" and "_rotated_270"
        if filename.startswith(idImage) and filename.endswith("_rotated_180.jpg"):
            image = Image.open(directory + filename)
            brightened_image = ImageEnhance.Brightness(image).enhance(1.25)
            # Save the modified image with a new filename
            new_filename = filename.split(".")[0] + "_brightened_1.25.jpg"  # add suffix "_brightened" before file extension
            brightened_image.save(directory + new_filename)

        if filename.startswith(idImage) and filename.endswith("_rotated_270.jpg"):
            image = Image.open(directory + filename)
            brightened_image = ImageEnhance.Brightness(image).enhance(1.25)
            # Save the modified image with a new filename
            new_filename = filename.split(".")[0] + "_brightened_1.25.jpg"  # add suffix "_brightened" before file extension
            brightened_image.save(directory + new_filename)

def brightened75test(directory):
        # Check if the filename matches the required pattern
        image = Image.open(directory)
        brightened_image = ImageEnhance.Brightness(image).enhance(0.75)
        return brightened_image
        
            

brightened75(directory,idImage)
brightened25(directory,idImage)

