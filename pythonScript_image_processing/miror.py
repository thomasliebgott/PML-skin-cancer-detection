import cv2

def miror(directory):
    # Load the image
    img = cv2.imread(directory)

    # Flip the image horizontally
    img_flip = cv2.flip(img, 1)

    # Save the flipped image
    return img_flip

directory = "D:\Hochschule\SS\PML\Project_PML\dx\AKIEDC\ISIC_0024329.jpg"
img_flip = miror(directory)

cv2.imwrite("flip_image.jpg", img_flip)
