import os
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from PIL import Image
import shutil 

def counterFile(path):
    
    countAKIEDC = 0
    # go to paht with AKIEDC at end
    for filename in os.listdir(os.path.join(path, 'AKIEDC')):
        # look if it exist 
        if os.path.isfile(os.path.join(path, 'AKIEDC', filename)):
            countAKIEDC += 1
    
    countBCC = 0
    for filename in os.listdir(os.path.join(path,'BCC')):
        
        if os.path.isfile(os.path.join(path,'BCC', filename)):
            countBCC += 1

    countBKL = 0
    for filename in os.listdir(os.path.join(path,'BKL')):
        
        if os.path.isfile(os.path.join(path,'BKL', filename)):
            countBKL += 1
    

    countDF = 0
    for filename in os.listdir(os.path.join(path,'DF')):
        
        if os.path.isfile(os.path.join(path,'DF', filename)):
            countDF += 1
    

    countMEL = 0
    for filename in os.listdir(os.path.join(path,'MEL')):
        
        if os.path.isfile(os.path.join(path,'MEL', filename)):
            countMEL += 1
    

    countNV = 0
    for filename in os.listdir(os.path.join(path,'NV')):
        
        if os.path.isfile(os.path.join(path,'NV', filename)):
            countNV += 1
    

    countVASC = 0
    for filename in os.listdir(os.path.join(path,'VASC')):
        
        if os.path.isfile(os.path.join(path,'VASC', filename)):
            countVASC += 1
    
    
    return countAKIEDC, countBCC, countBKL, countDF, countMEL, countNV, countVASC

def miror(directory):
    
    img = cv2.imread(directory)

    img_flip = cv2.flip(img, 1)

    return img_flip

def erosion(directory):
    erosion_size = 1 
    erosion_shape = cv2.MORPH_ELLIPSE
    
    element = cv2.getStructuringElement(erosion_shape ,(2 * erosion_size +1, 2* erosion_size +1),(erosion_size, erosion_size))

    img = cv2.imread(directory)
    
    dist = cv2.erode(img, element)
    
    return dist

def dilatation(directory):
    dilation_size = 1 
    dilation_shape = cv2.MORPH_ELLIPSE
    
    element = cv2.getStructuringElement(dilation_shape, (2 * dilation_size + 1, 2 * dilation_size + 1), (dilation_size, dilation_size))

    img = cv2.imread(directory)
    
    dist = cv2.dilate(img, element)
    
    return dist

def rotation90(directory):
    
    img = Image.open(directory)
    
    rotated_img = img.rotate(-90, expand=True)

    width, height = img.size
    aspect_ratio = width / height

    resized_img = rotated_img.resize((int(height * aspect_ratio), height))

    opencv_img = cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2BGR)
    
    return opencv_img

def rotation180(directory):

    img = cv2.imread(directory)
    
    rotated_img = cv2.rotate(img, cv2.ROTATE_180)

    return rotated_img

def rotation270(directory):
    img = Image.open(directory)
    
    rotated_img = img.rotate(90, expand=True)

    width, height = img.size
    aspect_ratio = width / height

    resized_img = rotated_img.resize((int(height * aspect_ratio), height))

    opencv_img = cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2BGR)
    
    return opencv_img

def brightened75(directory):
    img = Image.open(directory)

    brightened_img = ImageEnhance.Brightness(img).enhance(0.75)

    opencv_img = cv2.cvtColor(np.array(brightened_img), cv2.COLOR_RGB2BGR)

    return opencv_img

def brightened25(directory):
    img = Image.open(directory)

    brightened_img = ImageEnhance.Brightness(img).enhance(1.25)

    opencv_img = cv2.cvtColor(np.array(brightened_img), cv2.COLOR_RGB2BGR)

    return opencv_img   

def RemoveHair(directory):

    segmentationPath = 'D:/Hochschule/SS/PML/Project_PML/dataverse_files/HAM10000_segmentations_lesion_tschandl/'
    file_name = os.path.basename(directory)
    segmentationImageName = os.path.splitext(file_name)
    segmentationImagePath=  segmentationPath + segmentationImageName [0]+ '_segmentation.png'
    sementatedImage = cv2.imread(segmentationImagePath, 0) #read image as grayscale

    orginalImage = cv2.imread(directory, 0) #read image as grayscale
    img= orginalImage.copy()
    l= 255
    u= 56

    cv2.namedWindow('image') # make a window with name 'image'
    canny = cv2.Canny(img, l, u)
    #cv2.imshow('canny', canny)
    contours, hierarchy = cv2.findContours(canny, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    h, w= img.shape
    countourimage= np.zeros((h, w), dtype = np.uint8)

    if sementatedImage is None:
        sementatedImage=np.zeros((h, w), dtype = np.uint8)
    
    contoursNumber=len(contours)
    if contoursNumber > 0:
        counter=0

        for cont in contours:
            rect = cv2.minAreaRect(cont)
            rectangleArea= rect[1][0]*rect[1][1]
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            numpy_horizontal_concat = np.concatenate((img, countourimage), axis=1) # to display image side by side

            area= cv2.contourArea(cont)
            rect1 = cv2.boundingRect(cont)
            ratioArea= area/ (rect1[2]*rect1[3])
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

            if  ratioArea < 0.29 and bigSide > 33:
                #img = cv2.drawContours(img,[box],0,(255,255,255),3)
                #cont1= cv2.convexHull(contours[counter])
                cv2.drawContours(countourimage, contours, counter, (255,255,255), 9)
            #cv2.rectangle(img, pt1=(rect1[0],rect1[1]), pt2=(rect1[0]+rect1[2],rect1[1]+rect1[3]), color=(255,255,255), thickness=3)
            #
            #cv2.imshow('111', countourimage)
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
    # go into the input_dir
    for originalRoot, directorys, _ in os.walk(input_dir):
        # go on each directory of the directory
        for directory in directorys:
            input_dir_path = os.path.join(originalRoot, directory)
            output_dir_path = os.path.join(output_dir, directory)

            # Create if it doesnt exist
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)

            # coppy all the image form inpu to output directory
            for file in os.listdir(input_dir_path):
                if file.endswith('.jpg'):
                    input_filepath = os.path.join(input_dir_path, file)
                    output_filepath = os.path.join(output_dir_path, file)
                    shutil.copy(input_filepath, output_filepath)

            # Increment the number of images in the directory count (setup with original number in dx)
            dir_counter = len(os.listdir(output_dir_path))

            # calculate the number of images that we want pro directory
            image_factor = ((num_images - dir_counter) / dir_counter)
            
            while dir_counter < num_images:
                for file in os.listdir(output_dir_path):
                    if file.endswith('.jpg'):
                        input_filepath = os.path.join(output_dir_path, file) #copy all the images from original dir to the new one
                        
                        # coutner of number of time function is apply
                        function_counter = 0 
                        
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

                            # break if enough images 
                            if dir_counter >= num_images:
                                break

                    # break if enough images 
                    if dir_counter >= num_images:
                        break

                # break if enough images 
                if dir_counter >= num_images:
                    break

def getMax(countAKIEDC,countBCC,countBKL,countDF,countMEL,countNV,countVASC):
    nn = [countAKIEDC,countBCC,countBKL,countDF,countMEL,countNV,countVASC]
    max = np.max(nn)
    return max 

if __name__ == "__main__":
       
    #########################  
    # Blend validation images 
    #
    input_dir_val = r"dx6_imageRichtigVerteilt/val"
    output_dir_val = r"dx7_imageRichtigVerteiltBlend/val"
    
    countAKIEDC, countBCC, countBKL, countDF, countMEL, countNV, countVASC = counterFile(input_dir_val)
    print('File count - AKIEDC : ', countAKIEDC)
    print('File count - BCC : ', countBCC)
    print('File count - BKL : ', countBKL)
    print('File count - DF : ', countDF)
    print('File count - MEL : ', countMEL)
    print('File count - NV : ', countNV)
    print('File count - VASC : ', countVASC)

    # find the max to know much images have to be modify
    num_images = getMax(countAKIEDC, countBCC, countBKL, countDF, countMEL, countNV, countVASC)
    
    print("num_image = " + str(num_images))
    
    # list of function / methode to call
    functions = [RemoveHair, miror, rotation90, rotation180, rotation270, brightened75, brightened25]

    dir_counter = 0
    
    applyImageProcessing(input_dir_val,output_dir_val,num_images,functions)
    
    countAKIEDC, countBCC, countBKL, countDF, countMEL, countNV, countVASC = counterFile(output_dir_val)
    print('File count - AKIEDC : ', countAKIEDC)
    print('File count - BCC : ', countBCC)
    print('File count - BKL : ', countBKL)
    print('File count - DF : ', countDF)
    print('File count - MEL : ', countMEL)
    print('File count - NV : ', countNV)
    print('File count - VASC : ', countVASC)
    
    #########################  
    # blend Train
    #
    
    input_dir_train = r"dx6_imageRichtigVerteilt/train"
    output_dir_train = r"dx7_imageRichtigVerteiltBlend/train"
    
    countAKIEDC, countBCC, countBKL, countDF, countMEL, countNV, countVASC = counterFile(input_dir_train)
    print('File count - AKIEDC : ', countAKIEDC)
    print('File count - BCC : ', countBCC)
    print('File count - BKL : ', countBKL)
    print('File count - DF : ', countDF)
    print('File count - MEL : ', countMEL)
    print('File count - NV : ', countNV)
    print('File count - VASC : ', countVASC)

    num_images = getMax(countAKIEDC, countBCC, countBKL, countDF, countMEL, countNV, countVASC)
    
    print("num_image = " + str(num_images))
    
    functions = [RemoveHair, miror, rotation90, rotation180, rotation270, brightened75, brightened25]

    dir_counter = 0
    
    applyImageProcessing(input_dir_train,output_dir_train,num_images,functions)
    
    countAKIEDC, countBCC, countBKL, countDF, countMEL, countNV, countVASC = counterFile(output_dir_train)
    print('File count - AKIEDC : ', countAKIEDC)
    print('File count - BCC : ', countBCC)
    print('File count - BKL : ', countBKL)
    print('File count - DF : ', countDF)
    print('File count - MEL : ', countMEL)
    print('File count - NV : ', countNV)
    print('File count - VASC : ', countVASC)
