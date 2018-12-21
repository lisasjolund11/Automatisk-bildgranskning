from granskning import get_files
from granskning import image_processing
import os
import matplotlib.pyplot as plt 

image_file_names, dir1, dir2 = get_files.get_file_dir()


for item in image_file_names:
    image_path1 = os.path.join(dir1, item)
    image_path2 = os.path.join(dir2, item)
    output1 = os.path.join(r'Diff', item[:-4] + '_Medusa.png')
    output2 = os.path.join(r'Diff', item[:-4] + '_Sweco.png')

    imageA, imageB = image_processing.main(image_path1, image_path2)

    plt.imsave(output1, imageA)
    plt.imsave(output2, imageB)

print('Script succeded!!')
