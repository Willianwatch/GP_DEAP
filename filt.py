#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 10:05:56 2018

@author: kyle
"""

from skimage import measure
import cv2
import numpy as np

def mean(image):
    return cv2.blur(image, (3, 3))

def equalizeHist(image):
    return cv2.equalizeHist(image)

def normalization(image):
    image = np.float(image)
    minVal, maxVal, _, _ = cv2.minMaxLoc(image)
    image = 255 * (image - minVal) / (maxVal - minVal)
    return np.uint8(np.clip(image, 0, 255))

def erode(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.erode(image, kernel)

def dilate(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(image, kernel)

def sobel(image):
    sobel_x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    return cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

def lightEdge(image):
    output = cv2.Laplacian(image, cv2.CV_16S)
    return np.uint8(np.clip(output, 0, 255))

def darkEdge(image):
    output =cv2.Laplacian(image, cv2.CV_16S) + 255
    return np.uint8(np.clip(output, 0, 255))

def lightPixel(image):
    meanValue = np.mean(image)
    image[image<meanValue] = 0
    return image

def darkPixel(image):
    meanValue = np.mean(image)
    image[image>=meanValue] = 255
    return image

def largeArea(image):
    output = np.copy(image)
    bwImage = np.ones(image.shape, dtype=np.uint8)
    rowEqual = np.uint8(cv2.absdiff(image[1:, :], image[:-1, :]) == 0）* 255
    colEqual = np.uint8(cv2.absdiff(image[:, 1:]), image[:,:-1]) == 0) * 255
    bwImage[1:, :] = cv2.add(rowEqual, bwImage[1:, :])
    bwImage[:-1, :] = cv2.add(rowEqual, bwImage[:-1, :])
    bwImage[:, 1:] = cv2.add(colEqual, bwImage[:, 1:])
    bwImage[:, :-1] = cv2.add(colEqual, bwImage[:, :-1])
    labelImage = measure.label(image, neighbors=4, background=0)
    regionProps = measure.regionprops(labelImage)
    averagePixels = np.mean(region.area for region in regionProps)
    for region in regionProps:
        if region.area < averagePixels:
            for row, col in region.coords:
                output[row, col] = 255
    return output
        
def smallArea(image):
    output = np.copy(image)
    bwImage = np.ones(image.shape, dtype=np.uint8)
    rowEqual = np.uint8(cv2.absdiff(image[1:, :], image[:-1, :]) == 0）* 255
    colEqual = np.uint8(cv2.absdiff(image[:, 1:]), image[:,:-1]) == 0) * 255
    bwImage[1:, :] = cv2.add(rowEqual, bwImage[1:, :])
    bwImage[:-1, :] = cv2.add(rowEqual, bwImage[:-1, :])
    bwImage[:, 1:] = cv2.add(colEqual, bwImage[:, 1:])
    bwImage[:, :-1] = cv2.add(colEqual, bwImage[:, :-1])
    labelImage = measure.label(image, neighbors=4, background=0)
    regionProps = measure.regionprops(labelImage)
    averagePixels = np.mean(region.area for region in regionProps)
    for region in regionProps:
        if region.area >= averagePixels:
            for row, col in region.coords:
                output[row, col] = 255
    return output

def inversion(image):
    return cv2.subtract(255, image)

def logicalSum(image1, image2):
    return cv2.max(image1, image2)

def logicalProd(image1, image2):
    return cv2.min(image1, image2)

def algebraicSum(image1, image2):
    image1 = np.float(image1)
    image2 = np.float(image2)
    output = image1 + image2 - image1 * image2 / 255
    return np.uint8(np.clip(output, 0, 255))

def algebraicProd(image1, image2):
    image1 = np.float(image1)
    image2 = np.float(image2)
    output = image1 * image2 / 255
    return np.uint8(output)

def boundedSum(image1, image2):
    return cv2.add(image1, image2)

def boundedProd(image1, image2):
    temp = cv2.subtract(255, image2)
    return cv2.subtract(image1, temp)