import cv2
import numpy as np

# Get the Image
def imread(path):
    img = cv2.imread(path)
    return img


#To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
def modcrop(img, scale =3):
    # Check the image is grayscale
    if len(img.shape) ==3:
        h, w, _ = img.shape
        h = (h / scale) * scale
        w = (w / scale) * scale
        img = img[0:h, 0:w, :]
    else:
        h, w = img.shape
        h = (h / scale) * scale
        w = (w / scale) * scale
        img = img[0:h, 0:w]
    return img
    
def preprocess(path ,scale = 3):
    img = imread(path)

    label_ = modcrop(img, scale)
    
    bicbuic_img = cv2.resize(label_,None,fx = 1.0/scale ,fy = 1.0/scale, interpolation = cv2.INTER_CUBIC)# Resize by scaling factor
    input_ = cv2.resize(bicbuic_img,None,fx = scale ,fy=scale, interpolation = cv2.INTER_CUBIC)# Resize by scaling factor
    return input_, label_



