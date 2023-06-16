import cv2
import numpy as np
from matplotlib import pyplot as plt

def main():
    img = cv2.imread('D:\Hochschule\SS\PML\Project_PML\dx\BCC\ISIC_0026343.jpg')
    assert img is not None, "file could not be read, check with os.path.exists()"

    edges = cv2.Canny(img,100,200)
    
    # Créer un masque de contour en inversant les contours détectés
    mask = 255 - edges  
    
    # Appliquer le masque de contour à l'image d'origine pour supprimer les contours
    result = cv2.bitwise_and(img, img, mask=mask)
    
    plt.plot(1),plt.imshow(img[...,::-1])
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    
    plt.plot(2),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    
    plt.plot(3),plt.imshow(result[...,::-1])
    plt.title('Result Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    
    return 0

if __name__ == "__main__":
    main()
