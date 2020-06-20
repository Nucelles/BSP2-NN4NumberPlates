import cv2
import os
import numpy as np
from urllib.request import urlopen
from albumentations import Resize, Compose, BboxParams
from matplotlib import pyplot as plt
from tqdm import tqdm
from random import shuffle
import keras.backend as K
import tensorflow as tf


def showBoundingBox(image, boundingBox):
    copy = image.copy()
    a = 1
    for obj in boundingBox[1:]:
        x, y, w, h = obj
        # cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,255),2)
        cropped = copy[y:y + h, x:x + w]
        cv2.imwrite("LincensePlate_{}.png".format(a), cropped)
        a += 1

    x, y, w, h = boundingBox[0]
    image = image[y:y + h, x:x + w]

    cv2.imshow('result', image)
    cv2.imwrite("LicensePlate.png", image)
    cv2.waitKey()
    cv2.destroyAllWindows()



def getPlateInfo(txt, summary=False):
    # only works with ULFP-DATASET
    l = 1
    info = ["", [], []]
    for line in txt.readlines():
        if l == 7:
           info[0] = line[len("plate: "):-1]
        elif l == 8:
            info[1] = [int(x) for x in line[len("position_plate: "):-1].split(" ")]
        elif l >= 9 and l!= 15:
            info[2].append([int(x) for x in line[len("	char 1: "):-1].split(" ")])
        elif l==15:
            info[2].append([int(x) for x in line[len("	char 7: "):].split(" ")])
        l+= 1


    if summary:
        print("License Plate:" + info[0])
        print("LP_BoundingBox:", info[1])
        print("LP_Characters: ")
        for c in info[-1]:
            print(c)
    #print(info)
    return info


"""
The following 4 functions were taken from the Albumentations github example tutorial
link: https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/example_bboxes.ipynb
"""


def visualize_bbox(img, bbox, class_id, class_idx_to_name, trueAnnotations = None, color=(0, 255, 0), thickness=2):
    BOX_COLOR_TRUE = (255, 0, 0)
    BOX_COLOR_PRED = (0, 255, 0)
    TEXT_COLOR = (255, 255, 255)

    x_min, y_min, x_max, y_max = [int(i) for i in bbox]

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    class_name = class_idx_to_name[class_id]
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR_PRED, -1)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,TEXT_COLOR, lineType=cv2.LINE_AA)

    if True:

        x_min, y_min, x_max, y_max = [int(i) for i in trueAnnotations]

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=BOX_COLOR_TRUE, thickness=thickness)
        ((text_width, text_height), _) = cv2.getTextSize("TruePrediction", cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR_TRUE, -1)
        cv2.putText(img, "TruePrediction", (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    TEXT_COLOR, lineType=cv2.LINE_AA)

    return img


def visualize(annotations, category_id_to_name, trueAnnotations=None):
    img = annotations['image'].copy()
    for idx, bbox in enumerate(annotations['bboxes']):
        if True:
            img = visualize_bbox(img, bbox, annotations['category_id'][idx], category_id_to_name, trueAnnotations=trueAnnotations)
        else:
            img = visualize_bbox(img, bbox, annotations['category_id'][idx], category_id_to_name)
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.show()


def download_image(url):
    data = urlopen(url).read()
    data = np.frombuffer(data, np.uint8)
    data_image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data_image


def dataAugmentFormat(img, bbox):
    return {'image': img, 'bboxes': [bbox], "category_id": [1]}


def augment(aug):
    return Compose(aug, bbox_params=BboxParams(format='coco', label_fields=['category_id']))


# Activity A
def prepareDataset(path, numOfImgs, imgSize, imagesToUnpack=None, split = None):
    pairedData = []
    imageDirectory = os.listdir(path + "/images")

    if imagesToUnpack > numOfImgs:
        numToUnpack = len(imageDirectory)
        numOfImgs = len(imageDirectory)

    if imagesToUnpack == None:
        numToUnpack = len(imageDirectory)

    else:
        numToUnpack = imagesToUnpack



    for currImage in tqdm(range(numToUnpack)):
        # Prepare the X input
        imageName = imageDirectory[currImage]
        image = cv2.imread(path + "/images/" + imageName)

        label = imageName[:-3] + "txt"
        label = getPlateInfo(open(path + "/plates/" + label))[1]




        # Resize the image keeping the bbox location
        composedImage = dataAugmentFormat(image, label)
        resizedComposedImage = augment([Resize(p=1, height=imgSize[1], width=imgSize[0])])
        finalComposedImage = resizedComposedImage(**composedImage)

        x_min, y_min, w, h = finalComposedImage["bboxes"][0]

        normalizedLabel = [x_min/(imgSize[1]-1),  #imgSize = [512, 288, 3]
                           y_min/(imgSize[0]-1),
                           (x_min+w)/(imgSize[1]-1),
                           (y_min+h)/(imgSize[0]-1)]

        image = finalComposedImage["image"] / 255

        dataPair = [image, normalizedLabel]
        pairedData.append(dataPair)

    shuffle(pairedData)

    if split:
        splitSize = int(len(pairedData) * 0.8)
        pairedDataTrain = pairedData[:splitSize]
        pairedDataVal = pairedData[splitSize:numOfImgs]

        x_train = np.array([a[0] for a in pairedDataTrain])
        y_train = np.array([[i for i in a[1]] for a in pairedDataTrain])

        x_val = np.array([a[0] for a in pairedDataVal])
        y_val = np.array([[i for i in a[1]] for a in pairedDataVal])

        return x_train, y_train, x_val, y_val


    pairedDataTrain = pairedData[:numOfImgs]
    x_train = np.array([a[0] for a in pairedData])
    y_train = np.array([[i for i in a[1]] for a in pairedData])
    return x_train, y_train


def showPrediction(coposition, trueBBox):
    category_id_to_name = {1: "License Plate"}
    visualize(coposition, category_id_to_name, trueBBox)


def showModelGraphs(history):
    """
         Plotting a CNN Training Curve
        :param history: Training History
        :return:
        """
    # Plot the loss and accuracy curves for training and validation
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
    ax[0].legend(loc='best', shadow=True)
    ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
    ax[1].legend(loc='best', shadow=True)
    plt.show()


def dice_loss(y_true, y_pred):
  numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
  denominator = tf.reduce_sum(y_true + y_pred, axis=-1)

  return 1 - (numerator + 1) / (denominator + 1)


def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
    weight_a = alpha * (1 - y_pred) ** gamma * targets
    weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

    return (tf.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b

def focal_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        logits = tf.log(y_pred / (1 - y_pred))

        loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=0.25, gamma=2, y_pred=y_pred)

        return tf.reduce_mean(loss)
