from __future__ import division
from PIL import Image
import os
import numpy as np

current_dir= os.getcwd()

annotations_dir = current_dir + "/visDrone/VisDrone2019-DET-train/annotations"
all_files = os.listdir(annotations_dir)
newlines=[]
fIM = open(current_dir + "yolov3/data/visDroneAll.txt","w+")
for i in range(len(all_files)):
    f= open(all_files[i],"w+")
    annotationLines = open(current_dir+ "/visDrone/VisDrone2019-DET-train/annotations/"+all_files[i], "r")
  
    im_dir = current_dir+"visDrone/VisDrone2019-DET-train/images/"+all_files[i].rstrip(".txt")+".jpg"
    fIM.write(im_dir +"\n")
  
    im = Image.open(im_dir,"r")
    width, height = im.size
    fW = open("data/darknet53/"+all_files[i], "w+")
    for line in annotationLines:
        line = line.rstrip('\n')
        line = line.split(",") 
    
        x_top= float(line[0])
        y_top=float(line[1])
        w=float(line[2])
        h=float(line[3])

        x_center=np.absolute(np.float32((x_top+w/2)/width))
        y_center=np.absolute(np.float32((y_top-h/2)/height))
        newline = str(line[5]) + " " + str(np.float32(x_center)) + " " + str(np.float32(y_center)) + " " + str(np.float32(w/width)) + " " + str(np.float32(h/height))
        fW.write(newline + "\n")
    fW.close()
fIM.close()

fData = open(current_dir + "yolov3/data/visDrone.names","w+")
visDroneNames =  [ignored regions, pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor, others]
for i in range(len(visDroneNames)):
  fData.write(visDroneNames +"\n")
fData.close()

fData = open(current_dir + "yolov3/data/visDrone.data", "w+")
line1 = "classes = " + len(visDroneNames)
fData.write(line1 +"\n")
line2 = "train = " + current_dir + "/yolov3/data/visDroneAll.txt"
fData.write(line2 +"\n")
line3 = "valid = " + current_dir + "/yolov3/data/visDroneAll.txt"
fData.write(line3 +"\n")
line4 = "names = " + current_dir + "/yolov3/data/visDrone.names"
fData.write(line4 +"\n")
fData.close()
