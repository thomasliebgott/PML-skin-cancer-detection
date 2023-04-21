import os
import shutil

folder_path = "./dx"

if not os.path.exists(folder_path):
    # check if the file exist if it's not it create it 
    os.mkdir(folder_path)
    print("folder dx created")

# Create sub folder and check before if they already exist 
subfolders = ["MEL", "NV", "BCC", "AKIEDC", "BKL", "DF", "VASC"]
for subfolder in subfolders:
    subfolder_path = os.path.join(folder_path, subfolder)
    if not os.path.exists(subfolder_path):
        os.mkdir(subfolder_path)
        print(f"folder {subfolder} created")

with open('dataverse_files/HAM10000_metadata.txt', 'r') as file: #red metafile
    lines = file.readlines()
    for line in lines:
        parts = line.strip().split(',') #separate data
        dx = parts[2]  #read dx info
        image_name_data = parts[1] + '.jpg' #setupe the image name 
        image_path = "dataverse_files/HAM10000_images_part_1/" + image_name_data #setup the image file 
        if os.path.exists(image_path): #look if the image exist

            if parts[2] == 'mel':
                dest_path = 'D:/Hochschule/PML/Project_PML/dx/MEL/' + image_name_data
                if not os.path.exists(dest_path):
                    shutil.copy(image_path, dest_path)
                    print('mel done')

            elif parts[2] == 'nv':
                dest_path = 'D:/Hochschule/PML/Project_PML/dx/NV/' + image_name_data
                if not os.path.exists(dest_path):
                    shutil.copy(image_path, dest_path)
                    print('nv done')

            elif parts[2] == 'bcc':
                dest_path = 'D:/Hochschule/PML/Project_PML/dx/BCC/' + image_name_data
                if not os.path.exists(dest_path):
                    shutil.copy(image_path, dest_path)
                    print('bcc done')

            elif parts[2] == 'akiec':
                dest_path = 'D:/Hochschule/PML/Project_PML/dx/AKIEDC/' + image_name_data
                if not os.path.exists(dest_path):
                    shutil.copy(image_path, dest_path)
                    print('akiec done')

            elif parts[2] == 'bkl':
                dest_path = 'D:/Hochschule/PML/Project_PML/dx/BKL/' + image_name_data
                if not os.path.exists(dest_path):
                    shutil.copy(image_path, dest_path)
                    print('bkl done')

            elif parts[2] == 'df':                        
                dest_path = 'D:/Hochschule/PML/Project_PML/dx/DF/' + image_name_data
                if not os.path.exists(dest_path):
                    shutil.copy(image_path, dest_path)
                    print('df done')

            elif parts[2] == 'vasc':
                dest_path = 'D:/Hochschule/PML/Project_PML/dx/VASC/' + image_name_data
                if not os.path.exists(dest_path):
                    shutil.copy(image_path, dest_path)
                    print('vasc done')
        else : 
            print("image path" + image_path + "existiert nicht")
            
            



