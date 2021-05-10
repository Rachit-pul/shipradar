from cv2 import cv2
from PIL import Image
# import pytesseract
import matplotlib.pyplot as plt
import keras_ocr
import numpy as np

pipeline = keras_ocr.pipeline.Pipeline()


def TextRecognition(imgPath):
    image = cv2.imread(imgPath)
    OriginalImg = image

    crop_img1 = image[20:170, 1030:]  # Ship Information
    crop_img2 = image[250:320, 1030:]  # Target Information
    crop_img3 = image[935:1200, 0:200]  # Deep Sea
    crop_img4 = image[0:50, 50:100]  # Range
    crop_img5 = image[80:110, 150:220]  # Stabilized
    crop_img6 = image[80:110, 100:150]  # Mode

    cv2.imwrite("./images/shipInfo.png", crop_img1)
    cv2.imwrite("./images/targetInfo.png", crop_img2)
    cv2.imwrite("./images/deepsea.png", crop_img3)
    cv2.imwrite("./images/range.png", crop_img4)
    cv2.imwrite("./images/stablized.png", crop_img5)
    cv2.imwrite("./images/mode.png", crop_img6)

    images = [keras_ocr.tools.read(url) for url in [
        "./images/mode.png",
        "./images/range.png",
        "./images/stablized.png"
    ]
    ]

    prediction_groups = pipeline.recognize(images)

    fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
    for ax, image, predictions in zip(axs, images, prediction_groups):
        keras_ocr.tools.drawAnnotations(
            image=image, predictions=predictions, ax=ax)

    imgPath1 = "./images/shipInfo.png"
    imgPath2 = "./images/targetInfo.png"
    imgPath3 = "./images/stablized.png"
    imgPath4 = "./images/deepsea.png"
