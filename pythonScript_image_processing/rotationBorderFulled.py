import cv2
import numpy as np

# Charger l'image
image = cv2.imread("rotation/ISIC_0027303.jpg")

# Obtenir les dimensions de l'image
(h, w) = image.shape[:2]

# Trouver la couleur des bords de l'image
top = image[0, int(w/2)]
right = image[int(h/2), w-1]
bottom = image[h-1, int(w/2)]
left = image[int(h/2), 0]
color = np.median([top, right, bottom, left], axis=0)

# Ajouter une bordure à l'image avec la couleur trouvée
border_size = int(max(h, w) * 0.5)
border_image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=color)

# Calculer la matrice de rotation
center = (w // 2 + border_size, h // 2 + border_size)
M = cv2.getRotationMatrix2D(center, 66, 1.0)

# Appliquer la matrice de rotation à l'image
rotated = cv2.warpAffine(border_image, M, (border_image.shape[1], border_image.shape[0]))

# Recadrer l'image pour enlever la bordure ajoutée
rotated = rotated[border_size:-border_size, border_size:-border_size]

# Afficher l'image d'origine et l'image rotatée
cv2.imshow("Original", image)
cv2.imshow("Rotated", rotated)
cv2.waitKey(0)