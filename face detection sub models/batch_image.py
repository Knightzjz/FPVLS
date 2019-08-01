'''
import os  
import shutil 

path = '~/facenet/data/images/CASIA-v5'
new_path = '~/facenet/data/images/CASIA'

for root, dirs, files in os.walk(path):
    for i in range(len(files)):
        print(files[i])
        if (files[i][-3:] == 'jpg') or (files[i][-3:] == 'bmp') or (files[i][-3:] == 'png') or (files[i][-3:] == 'JPG'):
            file_path = root+'/'+files[i]  
            new_file_path = new_path+ '/'+ files[i]  
            shutil.copy(file_path,new_file_path)  
'''








import os
import shutil
print 'a'
path = '/home/knight/facenet/training_data/img_align_celeba'
new_path = '/home/knight/facenet/traning_data/others_celeba'
print 'path'
for root, dirs, files in os.walk(path):
    if len(dirs) == 0:
        for i in range(len(files)):
            if (files[i][-3:-1] == 'jpg') or (files[i][-3:-1] == 'png') or (files[i][-3:-1] == 'bmp'):
                file_path = root+'/'+files[i]
                new_file_path = new_path+ '/'+ files[i]
                shutil.move(file_path,new_file_path)
        print (i, "images are moved to new folder")