import os
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim
from scipy.signal import convolve2d
from scipy.ndimage import generic_filter
import argparse
import imutils 
import cv2
import numpy as np


# Show image function
def display_image(img):
    fig = plt.figure(figsize=(40,40))
    ax = fig.add_subplot(111)
    ax.imshow(img)

def main(fileA, fileB):
    imageA = cv2.imread(fileA)
    imageB = cv2.imread(fileB)
    thA, thB = preprocessing(imageA, imageB)
    diff = image_difference(thA, thB)
    diff_average = eight_neighbor_average(diff)
    diff_resample = loop_over_diff_average(diff_average)
    diff_resample = (diff_resample*255).astype('uint8')
    mark_diff(diff_resample, imageA, imageB)
    return imageA, imageB


def preprocessing(imageA, imageB):
    # Change from 3 color channels to one gray channel using opencv function
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)


    # Blur images using numpy
    kernelA = np.ones(shape=(1,1), dtype=np.float32)/25
    #kernelB = np.ones(shape=(1,1), dtype=np.float32)/25

    dstA = cv2.filter2D(grayA, -1, kernelA)
    #dstB = cv2.filter2D(grayB, -1, kernelB)

    thA = cv2.threshold(dstA, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thB = cv2.threshold(grayB, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    return thA, thB


def image_difference(thA, thB):
    # Calculate the difference between the two images usin comapre_ssim
    (score, diff) = compare_ssim(thA, thB, full=True)
    diff = (diff*255).astype('uint8')
    print('SSIM: {}'.format(score))

    return diff


def eight_neighbor_average(image_array):
    kernel = np.ones((7,7))
    kernel[1, 1] = 0
    neighbor_sum = convolve2d(image_array, kernel, mode='same',boundary='fill', fillvalue=0)

    num_neighbor = convolve2d(np.ones(image_array.shape), kernel, mode='same', boundary='fill', fillvalue=0)
    
    result = neighbor_sum / num_neighbor 

    return result

def loop_over_diff_average(image):
    rows,cols = image.shape
    for i in range(rows):
        for j in range(cols):
            if image[i,j] < 2:
                image[i, j] = 255
            else:
                image[i, j] = 0
            
    return image
    

def mark_diff(diff_resample, imageA, imageB):

    ret, thresh = cv2.threshold(diff_resample, 0, 255, cv2.THRESH_BINARY)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(imageA, (x,y), (x+w, y+h), color=(255,0,255), thickness=2)
        cv2.rectangle(imageB, (x,y), (x+w, y+h), color=(255,0,0), thickness=2)

    

