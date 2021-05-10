from cv2 import cv2
from PIL import Image
import matplotlib.pyplot as plt
import keras_ocr
import numpy as np
import tensorflow as tf
from collections import Iterable
import os, imutils
from datetime import datetime




def flatten(lis):
     for item in lis:
         if isinstance(item, Iterable) and not isinstance(item, str):
             for x in flatten(item):
                 yield x
         else:        
             yield item
detector = keras_ocr.detection.Detector()
recogo = keras_ocr.recognition.Recognizer(weights='kurapan')
pipeline = keras_ocr.pipeline.Pipeline(detector= detector, recognizer=recogo)
# imgPath = 'C:/Users/rachi/Desktop/shipradar/img.png'

def recognizer(imgPath):
    image = cv2.imread(imgPath)
    OriginalImg = image

    crop_img1 = image[20:170, 1030:]  # Ship Information
    thresh = cv2.threshold(crop_img1, 150, 255, cv2.THRESH_BINARY_INV)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    result = 255 - close 
    crop_imgg1 = cv2.GaussianBlur(result, (5,5), 0)

    crop_img2 = image[250:320, 1030:]  # Target Information
    thresh = cv2.threshold(crop_img1, 150, 255, cv2.THRESH_BINARY_INV)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    result = 255 - close 
    crop_imgg2 = cv2.GaussianBlur(result, (5,5), 0)
    imggg = Image.fromarray(crop_img2,'RGB')
    crop_img3 = image[935:1200, 0:200]  # Deep Sea
    thresh = cv2.threshold(crop_img1, 150, 255, cv2.THRESH_BINARY_INV)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    result = 255 - close 
    crop_imgg3 = cv2.GaussianBlur(result, (5,5), 0)
    crop_img4 = image[0:50, 50:100]  # Range
    thresh = cv2.threshold(crop_img1, 150, 255, cv2.THRESH_BINARY_INV)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    result = 255 - close 
    crop_imgg4 = cv2.GaussianBlur(result, (5,5), 0)
    crop_img5 = image[80:110, 150:220]  # Stabilized
    thresh = cv2.threshold(crop_img1, 150, 255, cv2.THRESH_BINARY_INV)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    result = 255 - close 
    crop_imgg5 = cv2.GaussianBlur(result, (5,5), 0)
    crop_img6 = image[80:110, 100:150]  # Mode
    thresh = cv2.threshold(crop_img1, 150, 255, cv2.THRESH_BINARY_INV)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    result = 255 - close 
    crop_imgg6 = cv2.GaussianBlur(result, (5,5), 0)

    cv2.imwrite("./static/shipInfo.png", crop_img1)
    # cv2.imwrite(".C:/Users/rachi/Desktop/shipradar/targetinfo.png", crop_img2)
    imggg.save('targetinfo.png')
    cv2.imwrite("./static/deepsea.png", crop_img3)
    cv2.imwrite("./static/range.png", crop_img4)
    cv2.imwrite("./static/stablized.png", crop_img5)
    cv2.imwrite("./static/mode.png", crop_img6)

    imageList = ["./static/shipInfo.png",
        # "./static/targetinfo.png",
        "./static/deepsea.png",
        "./static/range.png",
        "./static/stablized.png",
        "./static/mode.png",
    ]
    f = open("data.txt", "a")
    now = datetime.now()
    f.write('\n'+'\n'+'TIMESTAMP:'+str(now) + '\n')
    for impath in imageList:
        images = [keras_ocr.tools.read(url) for url in [
            # "C:/Users/rachi/Desktop/shipradar/mode.png",
            # "C:/Users/rachi/Desktop/shipradar/range.png",
            # "C:/Users/rachi/Desktop/shipradar/stablized.png",
            # "C:/Users/rachi/Desktop/shipradar/shipInfo.png"
            impath
        ]
        ]

        prediction_groups = pipeline.recognize(images)

        # for i in prediction_groups:
        #     if i==prediction_groups[0]:
        #         print(i[1][0])
        #     print(i[0][0])
        output =(list(flatten(prediction_groups)))
        dataToBeAdded = []
        for i in output:
            k=i
            if type(i) is str:
                print(i)
                if ( impath == './static/shipInfo.png'):
                    if(i.isnumeric()):
                        k = int(i) /10
                dataToBeAdded.append(str(k))
        # try:
        #     os.remove("data.txt")
        # except OSError:
        #     pass
        f = open("data.txt", "a")
        # if(impath== "./static/shipInfo.png"):
        counter = 0
        for y in dataToBeAdded:
            if counter ==0:
                f.write(impath)
                f.write('\n')
            f.write(y)
            f.write("\t")
            counter +=1
            if(counter%3==0):
                f.write("\n")
            if counter == len(dataToBeAdded):
                f.write('\n')

    # fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
    # for ax, image, predictions in zip(axs, images, prediction_groups):
    #     keras_ocr.tools.drawAnnotations(
    #         image=image, predictions=predictions, ax=ax)

    # imgPath1 = "./images/shipInfo.png"
    # imgPath2 = "./images/targetInfo.png"
    # imgPath3 = "./images/stablized.png"
    # imgPath4 = "./images/deepsea.png"