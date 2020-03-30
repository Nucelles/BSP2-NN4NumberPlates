import cv2
import os
import numpy as np
from urllib.request import urlopen
from albumentations import Resize, Compose, BboxParams
from matplotlib import pyplot as plt
from tqdm import tqdm
from random import shuffle


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
    # This only works with the UFPR-ALPR dataset, as it is saved in this format
    l = 1
    info = ["", [], []]
    for line in txt.readlines():
        if l == 7:
            info[0] = line[len("plate: "):-1]
        elif l == 8:
            info[1] = [int(x) for x in line[len("position_plate: "):-1].split(" ")]
        elif l >= 9 and l != 15:
            info[2].append([int(x) for x in line[len("	char 1: "):-1].split(" ")])
        elif l == 15:
            info[2].append([int(x) for x in line[len("	char 7: "):].split(" ")])
        l += 1

    if summary:
        print("License Plate:" + info[0])
        print("LP_BoundingBox:", info[1])
        print("LP_Characters: ")
        for c in info[-1]:
            print(c)
    # print(info)
    return info


"""
The following 4 functions were taken from the Albumentations github example tutorial
link: https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/example_bboxes.ipynb
"""


def visualize_bbox(img, bbox, class_id, class_idx_to_name, color=(255, 0, 0), thickness=2):
    BOX_COLOR = (255, 0, 0)
    TEXT_COLOR = (255, 255, 255)
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    class_name = class_idx_to_name[class_id]
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_COLOR,
                lineType=cv2.LINE_AA)
    return img


def visualize(annotations, category_id_to_name):
    img = annotations['image'].copy()
    for idx, bbox in enumerate(annotations['bboxes']):
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


def prepareDataset(path, numOfImgs, imgSize, imagesToUnpack=None):
    pairedData = []
    imageDirectory = os.listdir(path + "/images")

    if imagesToUnpack == None:
        numToUnpack = len(imageDirectory)
    else:
        numToUnpack = imagesToUnpack

    for currImage in tqdm(range(numToUnpack)):
        # Prepare the X input
        imageName = imageDirectory[currImage]
        image = cv2.imread(path + "/images/" + imageName)
        # = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = image / 255
        # print(image.shape)

        # Prepare the Y input
        label = imageName[:-3] + "txt"
        label = getPlateInfo(open(path + "/plates/" + label))[1]

        # Resize the image keeping the bbox location
        composedImage = dataAugmentFormat(image, label)
        resizedComposedImage = augment([Resize(p=1, height=imgSize[1], width=imgSize[0])])
        finalComposedImage = resizedComposedImage(**composedImage)

        dataPair = [finalComposedImage["image"], finalComposedImage["bboxes"][0]]
        pairedData.append(dataPair)

    shuffle(pairedData)
    pairedData = pairedData[:numOfImgs]
    x_train = np.array([a[0] for a in pairedData])
    y_train = np.array([[i / 255 for i in a[1]] for a in pairedData])
    return x_train, y_train


def showPrediction(coposition):
    category_id_to_name = {1: "License Plate"}
    visualize(coposition, category_id_to_name)
    # print('here')
