import cv2
import os 

def showBoundingBox(image, boundingBox):
    copy = image.copy()
    a = 1
    for obj in boundingBox[1:]:
        x, y, w, h = obj
        #cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,255),2)
        cropped = copy[y:y+h, x:x+w]
        cv2.imwrite("LincensePlate_{}.png".format(a), cropped)
        a+=1

    x, y, w, h = boundingBox[0]
    image = image[y:y+h, x:x+w]
    
    cv2.imshow('result', image)
    cv2.imwrite("LicensePlate.png",image)
    cv2.waitKey()
    cv2.destroyAllWindows()


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
    print(info)
    return info

# Change directory to folder containing labaels
os.chdir("C:/Users/Stella/Desktop/University/Semester 2/BSP S2/DATA/UFPR-ALPR dataset/DATA/testing/plates")
# Select the label
txt = "track0091[01].txt"

getPlateInfo(txt, False)


    
boundingBox = [[855, 504, 64, 21], [859, 510, 7, 12], [867, 510, 7, 12],
               [874, 511, 7, 11], [885, 511, 7, 11], [893, 511, 7, 11], [902, 511, 3, 12],
               [910, 511, 3, 11]]
image = cv2.imread('track0091[01].png',0)
#showBoundingBox(image, boundingBox)
