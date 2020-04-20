from __future__ import division
from PIL import Image
import os
import numpy as np
import glob
import shutil

annotations_dir = "VisDrone2019-DET-train/annotations/"
all_files = os.listdir(annotations_dir)
newlines=[]

fIM = open("YOLOv3/data/visDrone_train.txt","w+")
for i in range(len(all_files)):
    annotationLines = open("VisDrone2019-DET-train/annotations/"+all_files[i], "r")
  
    im_dir_train ="./visDrone/images/"+all_files[i].rstrip(".txt")+".jpg"
    fIM.write(im_dir_train +"\n")
  
    im_dir_source="VisDrone2019-DET-train/images/"+all_files[i].rstrip(".txt")+".jpg"
    im = Image.open(im_dir_source,"r")
    width, height = im.size

    fW = open("visDrone/labels/"+all_files[i], "w+")
    for line in annotationLines:
        line = line.rstrip('\n')
        line = line.split(",") 
    
        x_top= float(line[0])
        y_top=float(line[1])
        w=float(line[2])
        h=float(line[3])

        x_center=np.absolute(np.float32((x_top-h/2)/width))
        y_center=np.absolute(np.float32((y_top+w/2)/height))
        newline = str(line[5]) + " " + str(np.float32(x_center)) + " " + str(np.float32(y_center)) + " " + str(np.float32(w)/width) + " " + str(np.float32(h)/height)
        fW.write(newline + "\n")
    fW.close()
fIM.close()

src_dir = "VisDrone2019-DET-train/images/"
dst_dir = "visDrone/images"
for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
    shutil.copy(jpgfile, dst_dir)

annotations_dir_val = "VisDrone2019-DET-val/annotations/"
all_files_val = os.listdir(annotations_dir_val)
newlines=[]

fIM = open("YOLOv3/data/visDrone_val.txt","w+")
for i in range(len(all_files_val)):
    annotationLines_val = open("VisDrone2019-DET-val/annotations/"+all_files_val[i], "r")
  
    im_dir_val ="./visDrone_val/images/"+all_files_val[i].rstrip(".txt")+".jpg"
    fIM.write(im_dir_val +"\n")
  
    im_dir_source="VisDrone2019-DET-val/images/"+all_files_val[i].rstrip(".txt")+".jpg"
    #im_dir_destination="visDrone/images/"
    im = Image.open(im_dir_source,"r")
    width, height = im.size
    #im_dir_dest = shutil.move(im_dir_source, im_dir_destination) 

    fW = open("visDrone_val/labels/"+all_files[i], "w+")
    for line in annotationLines:
        line = line.rstrip('\n')
        line = line.split(",") 
    
        x_top= float(line[0])
        y_top=float(line[1])
        w=float(line[2])
        h=float(line[3])

        x_center=np.absolute(np.float32((x_top-h/2)/width))
        y_center=np.absolute(np.float32((y_top+w/2)/height))
        newline = str(line[5]) + " " + str(np.float32(x_center)) + " " + str(np.float32(y_center)) + " " + str(np.float32(w)/width) + " " + str(np.float32(h)/height)
        fW.write(newline + "\n")
    fW.close()
fIM.close()

src_dir = "VisDrone2019-DET-val/images/"
dst_dir = "visDrone_val/images"
for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
    shutil.copy(jpgfile, dst_dir)

fD = open("YOLOv3/data/visDrone.data", "w+")
fD.write( "classes=12"+ "\n")
fD.write( "train=YOLOv3/data/visDrone_train.txt"+ "\n")
fD.write( "valid=YOLOv3/data/visDrone_val.txt"+ "\n")
fD.write( "names=YOLOv3/data/visDrone.names"+ "\n")
fD.close()

fN = open("YOLOv3/data/visDrone.names", "w+")
names = ["ignored regions(0)", 
         "pedestrian(1)", 
         "people(2)", 
         "bicycle(3)", 
         "car(4)", 
         "van(5)", 
         "truck(6)", 
         "tricycle(7)", 
         "awning-tricycle(8)", 
         "bus(9)", 
         "motor(10)", 
         "others(11)"]
for i in range(len(names)):
    fN.write(str(names[i]) + "\n")
fN.close()
