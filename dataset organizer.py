import os
datasets = ["validation", "testing", "training"]
path = os.getcwd()


# HIERARCHY
# DATA -> Training -> Images
#                     Plates
#         Validation -> Images
#                       Plates
#         Testing -> Images
#                    Plates

for split in datasets:
    
    directory = os.listdir(path+"/"+split)
    print("Current Dataset:" + split)

    for folder in directory:
        
        currPath = path+"\\{}\\".format(split)+folder
        
        print("Current Folder:" + folder)

        items = os.listdir(currPath)
        
        for item in items:
            if ".png" in item:
                os.rename(currPath+"/"+item, "DATA/"+split+"/images/"+item)
            else:
                os.rename(currPath+"/"+item, "DATA/"+split+"/plates/"+item)

    pngs = os.listdir(os.getcwd()+"/DATA/"+split+"/images")
    txts = os.listdir(os.getcwd()+"/DATA/"+split+"/plates")
    notIncluded = []
    a,b = 0,0

    for t in txts:
        equivalentPNG = t[:-4]+".png"
        if equivalentPNG in pngs:
            #print(equivalentPNG+" Found!")
            a += 1
        else:
            b += 1
            #print(equivalentPNG+" Not Found!")
            notIncluded.append(t)

    print("Summary of {}\nMatched{}\nUnmatched{}".format(split, a, b))
    if notIncluded != []:
        print(notIncluded)
