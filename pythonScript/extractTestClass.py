import os
import shutil

folder_path= r"dx4/test"

if not os.path.exists(folder_path):
    # check if the file exist if it's not it create it 
    os.mkdir(folder_path)
    print("folder dx4 created")

# create sub folder and check before if they already exist 
subfolders = ["MEL", "NV", "BCC", "AKIEDC", "BKL", "DF", "VASC"]

for subfolder in subfolders:
    subfolder_path_img = os.path.join(folder_path, subfolder)
    if not os.path.exists(subfolder_path_img):
        os.mkdir(subfolder_path_img)
        print(f"folder {subfolder} created")

with open(r'dataverse_files\ISIC2018_Task3_Test_GroundTruth.csv', 'r') as file: #red metafile
    lines = file.readlines()
    for line in lines:
        parts_data = line.strip().split(',') #separate data
        dx = parts_data[2]  #read dx info
        image_name_data = parts_data[1] + '.jpg' #setup the image name 
        image_path = r"dataverse_files\ISIC2018_Task3_Test_Images\ISIC2018_Task3_Test_Images/" + image_name_data
        if os.path.exists(image_path) : #look if the image exist
            if parts_data[2] == 'mel':
                dest_path = r'dx4\\test/MEL/' + image_name_data
                if not os.path.exists(dest_path):
                    shutil.copy(image_path, dest_path)
                    print('mel done')

            elif parts_data[2] == 'nv':
                dest_path = r'dx4\\test/NV/' + image_name_data
                if not os.path.exists(dest_path):
                    shutil.copy(image_path, dest_path)
                    print('nv done')

            elif parts_data[2] == 'bcc':
                dest_path = r'dx4\\test/BCC/' + image_name_data
                if not os.path.exists(dest_path):
                    shutil.copy(image_path, dest_path)
                    print('ncc done')

            elif parts_data[2] == 'akiec':
                dest_path = r'dx4\\test/AKIEDC/' + image_name_data
                if not os.path.exists(dest_path):
                    shutil.copy(image_path, dest_path)
                    print('akiec done')

            elif parts_data[2] == 'bkl':
                dest_path = r'dx4\\test/BKL/' + image_name_data
                if not os.path.exists(dest_path):
                    shutil.copy(image_path, dest_path)
                    print('bkl done')

            elif parts_data[2] == 'df':                        
                dest_path = r'dx4\\test/DF/' + image_name_data
                if not os.path.exists(dest_path):
                    shutil.copy(image_path, dest_path)
                    print('df done')

            elif parts_data[2] == 'vasc':
                dest_path = r'dx4\\test/VASC/' + image_name_data
                if not os.path.exists(dest_path):
                    shutil.copy(image_path, dest_path)
                    print('vasc done')

        else : 
            print("image path" + image_path + "do not exist")
            
