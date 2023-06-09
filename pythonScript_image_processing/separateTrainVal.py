import os
import random
import shutil

# input folder with images already separated
input_folder = r'dx'

# paht for training und validation
output_train_folder = r'dx6/train'
output_val_folder = r'dx6/val'

# train images pourcentage
train_percentage = 0.8

# Parcourir chaque sous-dossier dans le dossier principal
for root, dirs, files in os.walk(input_folder):
    for dir_name in dirs:
        dir_path = os.path.join(root, dir_name)
        
        # Liste de toutes les images dans le sous-dossier actuel
        images = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        
        # Nombre d'images à utiliser pour l'entraînement et la validation
        num_train = int(len(images) * train_percentage)
        num_val = len(images) - num_train
        
        # Mélanger aléatoirement la liste des images
        random.shuffle(images)
        
        # Diviser les images en images d'entraînement et de validation
        train_images = images[:num_train]
        val_images = images[num_train:]
        
        # Créer les dossiers de sortie pour les images d'entraînement et de validation
        output_train_dir = os.path.join(output_train_folder, dir_name)
        output_val_dir = os.path.join(output_val_folder, dir_name)
        os.makedirs(output_train_dir, exist_ok=True)
        os.makedirs(output_val_dir, exist_ok=True)
        
        # Déplacer les images d'entraînement vers le dossier de sortie correspondant
        for image in train_images:
            src = os.path.join(dir_path, image)
            dst = os.path.join(output_train_dir, image)
            shutil.copy(src, dst)
        
        # Déplacer les images de validation vers le dossier de sortie correspondant
        for image in val_images:
            src = os.path.join(dir_path, image)
            dst = os.path.join(output_val_dir, image)
            shutil.copy(src, dst)

print("Séparation des images terminée.")
