import os
import argparse
import cv2
import glob

def get_file_dir():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--first", required=True,
    help="first input directory Medusa")
    ap.add_argument("-s", "--second", required=True,
    help="second input directory Sweco")
    args = vars(ap.parse_args())

    dir1 = args['first']
    dir2 = args['second']

    filenames1 = []
    filenames2 = []

    for file1 in os.listdir(dir1):
        filename1 = os.path.basename(file1)
        filenames1.append(filename1)
        

    for file2 in os.listdir(dir2):
        filename2 = os.path.basename(file2)
        if filename2 in filenames1:
            filenames2.append(filename2)
    

    return filenames2, dir1, dir2

    