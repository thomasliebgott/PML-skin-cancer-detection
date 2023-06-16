import os
import shutil

folder_path = r"dx_OriginalVerteiltImages"

if not os.path.exists(folder_path):
    # look if the fixe exist and if not create it 
    os.mkdir(folder_path)
    print("folder dx created")

# look if subfolder exist for each classe and if not create it
subFolders_name = ["MEL", "NV", "BCC", "AKIEDC", "BKL", "DF", "VASC"]

for subFolder in subFolders_name:
    subFolder_path = os.path.join(folder_path, subFolder)
    if not os.path.exists(subFolder_path):
        os.mkdir(subFolder_path)
        print(f"folder {subFolder} created")

input_file = r'dataverse_files/HAM10000_metadata.txt'
with open(input_file, 'r') as file: #red metafile
    lines_file = file.readlines()
    for line in lines_file:
        parts_data = line.strip().split(',') #separate data
        dx = parts_data[2]  #read dx info (type of cancer/classe)
        
        image_name_data = parts_data[1] + '.jpg' #setup the image name with the name give in the file
        
        image_path = r"dataverse_files/HAM10000_images_part_1/" + image_name_data #setup the image file 
        image_path_2 = r"dataverse_files/HAM10000_images_part_2/" + image_name_data

        if os.path.exists(image_path) or os.path.exists(image_path_2) : #look if the image exist
            if parts_data[2] == 'mel':
                dest_path = 'dx/MEL/' + image_name_data
                if not os.path.exists(dest_path):
                    if os.path.exists(image_path):
                        shutil.copy(image_path, dest_path)
                        print('mel done (1)')
                    elif os.path.exists(image_path_2):
                        shutil.copy(image_path_2, dest_path)
                        print('mel done (2)')

            elif parts_data[2] == 'nv':
                dest_path = 'dx/NV/' + image_name_data
                if not os.path.exists(dest_path):
                    if os.path.exists(image_path):
                        shutil.copy(image_path, dest_path)
                        print('nv done (1)')
                    elif os.path.exists(image_path_2):
                        shutil.copy(image_path_2, dest_path)
                        print('nv done (2)')

            elif parts_data[2] == 'bcc':
                dest_path = 'dx/BCC/' + image_name_data
                if not os.path.exists(dest_path):
                    if os.path.exists(image_path):
                        shutil.copy(image_path, dest_path)
                        print('bcc done (1)')
                    elif os.path.exists(image_path_2):
                        shutil.copy(image_path_2, dest_path)
                        print('bcc done (2)')

            elif parts_data[2] == 'akiec':
                dest_path = 'dx/AKIEDC/' + image_name_data
                if not os.path.exists(dest_path):
                    if os.path.exists(image_path):
                        shutil.copy(image_path, dest_path)
                        print('akiec done (1)')
                    elif os.path.exists(image_path_2):
                        shutil.copy(image_path_2, dest_path)
                        print('akiec done (2)')

            elif parts_data[2] == 'bkl':
                dest_path = 'dx/BKL/' + image_name_data
                if not os.path.exists(dest_path):
                    if os.path.exists(image_path):
                        shutil.copy(image_path, dest_path)
                        print('bkl done (1)')
                    elif os.path.exists(image_path_2):
                        shutil.copy(image_path_2, dest_path)
                        print('bkl done (2)')
            elif parts_data[2] == 'df':                        
                dest_path = 'dx/DF/' + image_name_data
                if not os.path.exists(dest_path):
                    if os.path.exists(image_path):
                        shutil.copy(image_path, dest_path)
                        print('df done (1)')
                    elif os.path.exists(image_path_2):
                        shutil.copy(image_path_2, dest_path)
                        print('df done (2)')

            elif parts_data[2] == 'vasc':
                dest_path = 'dx/VASC/' + image_name_data
                if not os.path.exists(dest_path):
                    if os.path.exists(image_path):
                        shutil.copy(image_path, dest_path)
                        print('vasc done (1)')
                    elif os.path.exists(image_path_2):
                        shutil.copy(image_path_2, dest_path)
                        print('vasc done (2)')

        else : 
            print("image path" + image_path + "do not exist")
            
            



