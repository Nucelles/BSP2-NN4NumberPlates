import cv2
import os
import numpy as np
from urllib.request import urlopen
from albumentations import Resize, Compose, BboxParams
from matplotlib import pyplot as plt



def showBoundingBox(image, boundingBox):
    copy = image.copy()
    a = 1
    for obj in boundingBox[1:]:
        x, y, w, h = obj
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cropped = copy[y:y+h, x:x+w]
        cv2.imwrite("LincensePlate_{}.png".format(a), cropped)
        a+=1

    x, y, w, h = boundingBox[0]
    image = image[y:y+h, x:x+w]
    
    cv2.imshow('result', image)
    #cv2.imwrite("LicensePlate.png",image)
    cv2.waitKey()
    cv2.destroyAllWindows()

def cocoToPascal(box):
    min_x, min_y, w, h = box
    newBox = [min_x, min_y, min_x + w, min_y + h]

    return newBox

def getPlateInfo(txt, summary):
    # read txt file
    # Save line 7 lincense plate
    # Save line 8 lincense plate BB
    # Save Char 1 through 7
    f = open(txt, "r")
    l = 1
    info = ["", [], []]
    for line in f.readlines():
        if l == 7:
           info[0] = line[len("plate: "):-1]
        elif l == 8:
            info[1] = cocoToPascal([int(x) for x in line[len("position_plate: "):-1].split(" ")])
        elif l >= 9 and l!= 15:
            info[2].append(cocoToPascal([int(x) for x in line[len("	char 1: "):-1].split(" ")]))
        elif l==15:
            info[2].append(cocoToPascal([int(x) for x in line[len("	char 7: "):].split(" ")]))
        l+= 1


    if summary:
        print("License Plate:" + info[0])
        print("LP_BoundingBox:", info[1])
        print("LP_Characters: ")
        for c in info[-1]:
            print(c)
    print(info)
    return info

def dataAugmentFormat(img, bbox, defaultID = [1]):
    return {'image': img, 'bboxes':[bbox], "category_id": defaultID}

def augment(aug):
    return Compose(aug, bbox_params=BboxParams(format='pascal_voc', label_fields=['category_id']))


def showImage(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None

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
            img = visualize_bbox(img, bbox, annotations['category_id'][idx], category_id_to_name, trueAnnotations)
        else:
            img = visualize_bbox(img, bbox, annotations['category_id'][idx], category_id_to_name)
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.show()




# Enter correct directory and get file name
path = "C:/Users/Nucelles 3.0/Documents/BICS/BSP S2/Project/DATA/training/"
os.chdir(path)
name = os.listdir(path+"images")[879][:-4]

# Read txt
bbox = getPlateInfo("plates/"+name+".txt", False)
print(bbox)
trueBBox = [757, 474, 898, 519]

# Read image
image = cv2.imread(path+"/images/"+name+'.png')

category_id_to_name = {1:"License Plate"}

dataExample = dataAugmentFormat(image, bbox[1])
visualize(dataExample, category_id_to_name, trueBBox)

dataExampleResized = augment([Resize(p=1, height=360, width=640)])
dataExampleResized = dataExampleResized(**dataExample)
visualize(dataExampleResized, category_id_to_name, trueBBox)


