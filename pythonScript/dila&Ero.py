from __future__ import print_function
import cv2 

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

if __name__ == "__main__":
    directory = cv2.imread("D:\Hochschule\SS\PML\Project_PML\dx\BKL\ISIC_0024409.jpg")
    dist = dilatation(directory)
    cv2.imwrite("dilated_image.jpg", dist)
    dist = erosion(directory)
    cv2.imwrite("eroded_image.jpg", dist)
