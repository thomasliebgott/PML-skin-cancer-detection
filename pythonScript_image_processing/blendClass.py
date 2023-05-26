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

def RemoveHair(directory):


    segmentationPath ='D:/pml/ali_branch/PML/dataverse_files/HAM10000_segmentations_lesion_tschandl/HAM10000_segmentations_lesion_tschandl/'
    file_name = os.path.basename(directory)
    segmentationImageName = os.path.splitext(file_name)
    segmentationImagePath=  segmentationPath + segmentationImageName [0]+ '_segmentation.png'
    sementatedImage = cv2.imread(segmentationImagePath, 0) #read image as grayscale

    orginalImage = cv2.imread(directory, 0) #read image as grayscale
    img= orginalImage.copy()
    l= 255
    u= 56

    #cv2.namedWindow('image') # make a window with name 'image'
    canny = cv2.Canny(img, l, u)
    #cv2.imshow('canny', canny)
    contours, hierarchy = cv2.findContours(canny, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    h, w= img.shape
    countourimage= np.zeros((h, w), dtype = np.uint8)

    if sementatedImage is None:
        sementatedImage=np.zeros((h, w), dtype = np.uint8)
    if len(contours) > 0:
        counter=0

        for cont in contours:
            rect = cv2.minAreaRect(cont)
            rectangleArea= rect[1][0]*rect[1][1]
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            #numpy_horizontal_concat = np.concatenate((img, countourimage), axis=1) # to display image side by side

            #area= cv2.contourArea(cont)
            #rect1 = cv2.boundingRect(cont)
            if rectangleArea==0 or (rect[1][0]==0 and rect[1][1]==0):
                counter+=1
                continue
            ratioRectSide= 0

            bigSide= 0
            if rect[1][0] > rect[1][1]:
                ratioRectSide= rect[1][1]/ rect[1][0]
                bigSide= rect[1][0]
            else:
                ratioRectSide= rect[1][0]/ rect[1][1]
                bigSide= rect[1][1]

            if  ratioRectSide < 0.3 and bigSide > 33:
                img = cv2.drawContours(img,[box],0,(255,255,255),3)
                cv2.drawContours(countourimage, contours, counter, (255,255,255), 9)
                #cv2.waitKey(1)
            counter+=1

    noHair= filterHair(orginalImage, countourimage, sementatedImage)
    noHair= cv2.cvtColor(noHair,cv2.COLOR_GRAY2RGB)
    #img= cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    #numpy_horizontal_concat = np.concatenate((img, noHair), axis=1) # to display image side by side
    #cv2.imshow('image', numpy_horizontal_concat)
    #cv2.waitKey(1)
    return noHair

def filterHair(orginal, hairMask, sementatedImage):
    w, h = orginal.shape[:2]
    noHair=orginal.copy()
    filterSize=9

    for i in range(0, w):
        for j in range(0, h):
            pixel=0
            if (hairMask[i, j] == 255):
                for k in range(i-int(filterSize/2), i+int(filterSize/2)):
                    for l in range(j-int(filterSize/2), j+ int(filterSize/2)):
                        if ( k >= 0 and l >= 0 and k < w and l < h and orginal[k, l] < 180):
                             pixel=max(pixel, orginal[k, l])

            if (pixel!= 0 and pixel!=255 and sementatedImage[i, j]!= 255):
                noHair[i, j]=pixel
            
    return noHair

def applyImageProcessing(input_dir, output_dir, num_images, functions):
    # Loop in the input directories
    for originalRoot, directorys, _ in os.walk(input_dir):
        # Loop in the directories in the input directory
        for directory in directorys:
            input_dir_path = os.path.join(originalRoot, directory)
            output_dir_path = os.path.join(output_dir, directory)

            # Create the output directory if it does not exist
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)

            # Copy all the images from the input directory to the output directory
            for file in os.listdir(input_dir_path):
                if file.endswith('.jpg'):
                    input_filepath = os.path.join(input_dir_path, file)
                    output_filepath = os.path.join(output_dir_path, file)
                    shutil.copy(input_filepath, output_filepath)

            # Increment the number of images in the directory count (setup with original number in dx)
            dir_counter = len(os.listdir(output_dir_path))

            #calculate the number of images that we want per directory
            image_factor = ((num_images - dir_counter) / dir_counter)
            
            while dir_counter < num_images:
                for file in os.listdir(output_dir_path):
                    if file.endswith('.jpg'):
                        input_filepath = os.path.join(output_dir_path, file) #copy all the images from original dir dx to our new dx2
                        
                        # Increment the number of time we apply function in an image
                        function_counter = 0 
                        # Increment the number of images in the directory count
                        image_count = 0    
                                              
                        while image_count < image_factor :
                            
                            if function_counter > len(functions)-1: #verifiy number of images to spread the number of operations and have the same number for each image 
                                function_counter = 0 
                            
                            func = functions[function_counter] #apply the function to the images
                            output_image = func(input_filepath)
                            output_filename = os.path.splitext(file)[0] +  '_' + str(image_count) + "_" + func.__name__ + os.path.splitext(file)[1] #set the image name with the function name  
                            output_filepath = os.path.join(output_dir_path, output_filename) #set the output file to dir in dx2 the new dir with image modifications
                            cv2.imwrite(output_filepath, output_image) #save the new image 

                            function_counter += 1 
                            
                            image_count += 1
                             
                            dir_counter += 1

                            # Break the loop if we have generated enough images
                            if dir_counter >= num_images:
                                break

                    # Break the loop if we have generated enough images
                    if dir_counter >= num_images:
                        break

                # Break the loop if we have generated enough images
                if dir_counter >= num_images:
                    break

def getMax(countAKIEDC,countBCC,countBKL,countDF,countMEL,countNV,countVASC):
    nn = [countAKIEDC,countBCC,countBKL,countDF,countMEL,countNV,countVASC]
    max = np.max(nn)
    return max 

if __name__ == "__main__":
    
    # Define the input and output directories
    input_dir = r"D:/pml/ali_branch/PML/dx"
    output_dir = r"D:/pml/ali_branch/PML/MitSegmentierungMaske"
    
    countAKIEDC, countBCC, countBKL, countDF, countMEL, countNV, countVASC = counterFile(input_dir)
    print('File count - AKIEDC : ', countAKIEDC)
    print('File count - BCC : ', countBCC)
    print('File count - BKL : ', countBKL)
    print('File count - DF : ', countDF)
    print('File count - MEL : ', countMEL)
    print('File count - NV : ', countNV)
    print('File count - VASC : ', countVASC)

    # Define the number of images to generate
    num_images = getMax(countAKIEDC, countBCC, countBKL, countDF, countMEL, countNV, countVASC)
    
    print("num_image = " + str(num_images))
    
    # Define a list of the functions
    functions = [RemoveHair, miror, rotation90, rotation180, rotation270, brightened75, brightened25]

    # Loop over the input images and apply the functions to generate new images
    dir_counter = 0
    
    applyImageProcessing(input_dir,output_dir,num_images,functions)
    
    countAKIEDC, countBCC, countBKL, countDF, countMEL, countNV, countVASC = counterFile(output_dir)
    print('File count - AKIEDC : ', countAKIEDC)
    print('File count - BCC : ', countBCC)
    print('File count - BKL : ', countBKL)
    print('File count - DF : ', countDF)
    print('File count - MEL : ', countMEL)
    print('File count - NV : ', countNV)
    print('File count - VASC : ', countVASC)
    
    
    
    