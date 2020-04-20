#!/usr/bin/env python
# coding: utf-8

# In[17]:


from __future__ import division
from PIL import Image
import os
import numpy as np
import glob
import shutil


# In[ ]:


current_dir= os.getcwd()
dir_Train_images="visDrone/images"
dir_Train_labels="visDrone/labels"
path_Train_images = os.path.join(current_dir, dir_Train_images)
os.mkdir(path_Train_images) 
path_Train_labels = os.path.join(current_dir, dir_Train_labels)
os.mkdir(path_Train_labels)

dir_Validation_images="visDrone_val/images"
dir_Validation_labels="visDrone_val/labels"
path_Validation_images = os.path.join(current_dir, dir_Validation_images)
os.mkdir(path_Validation_images)
path_Validation_labels = os.path.join(current_dir, dir_Validation_labels)
os.mkdir(path_Validation_labels)


# In[15]:


annotations_dir = "VisDrone2019-DET-train/annotations/"
all_files = os.listdir(annotations_dir)
newlines=[]

fIM = open("YOLOv3/data/visDrone_train.txt","w+")
for i in range(len(all_files)):
    annotationLines = open("/VisDrone2019-DET-train/annotations/"+all_files[i], "r")
  
    im_dir_train ="visDrone/images/"+all_files[i].rstrip(".txt")+".jpg"
    fIM.write(im_dir_train +"\n")
  
    im_dir_source="VisDrone2019-DET-train/images/"+all_files[i].rstrip(".txt")+".jpg"
    #im_dir_destination="visDrone/images/"
    im = Image.open(im_dir_source,"r")
    width, height = im.size
    #im_dir_dest = shutil.move(im_dir_source, im_dir_destination) 

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


# In[18]:


src_dir = "VisDrone2019-DET-train/images/"
dst_dir = "visDrone/images"
for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
    shutil.copy(jpgfile, dst_dir)


# In[23]:


annotations_dir_val = "VisDrone2019-DET-val/annotations/"
all_files_val = os.listdir(annotations_dir_val)
newlines=[]

fIM = open("YOLOv3/data/visDrone_val.txt","w+")
for i in range(len(all_files_val)):
    annotationLines_val = open("VisDrone2019-DET-val/annotations/"+all_files_val[i], "r")
  
    im_dir_val ="visDrone_val/images/"+all_files_val[i].rstrip(".txt")+".jpg"
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


# In[24]:


src_dir = "VisDrone2019-DET-val/images/"
dst_dir = "visDrone_val/images"
for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
    shutil.copy(jpgfile, dst_dir)


# In[ ]:


fD = open("YOLOv3/data/visDrone.data", "w+")
fD.write( "classes=12"+ "\n")
fD.write( "train=YOLOv3/data/visDrone_train.txt"+ "\n")
fD.write( "valid=YOLOv3/data/visDrone_val.txt"+ "\n")
fD.write( "names=YOLOv3/data/visDrone.names"+ "\n")
fD.close()


# In[ ]:


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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:


annotations_dir = "VisDrone/VisDrone2019-DET-train/annotations"
all_files = os.listdir(annotations_dir)
newlines=[]
fIM = open("data/data_to_train.txt","w+")
for i in range(len(all_files)):
    print(str(i) + "\t"+all_files[i])  
    im_dir = "VisDrone/VisDrone2019-DET-train/images/"+all_files[i].rstrip(".txt")+".jpg"
    fIM.write("/content/gdrive/My Drive/coco/images/"+all_files[i].rstrip(".txt")+".jpg" +"\n")


# In[6]:


annotations_dir = "visDrone/VisDrone2019-DET-train/annotations"
all_files = os.listdir(annotations_dir)
newlines=[]
fIM = open("data/data_to_train.txt","w+")
for i in range(len(all_files)):
    #f= open(all_files[i],"w+")
    #annotationLines = open( "visDrone/VisDrone2019-DET-train/annotations/"+all_files[i], "r")
  
    im_dir ="/content/gdrive/My Drive/YOLOv3/coco/images/"+all_files[i].rstrip(".txt")+".jpg"
    fIM.write(im_dir +"\n")


# In[33]:


current_dir= os.getcwd()
annotations_dir = "data/darknet53"
all_files = os.listdir(annotations_dir)
bbox= []
Bbox = {}
count=0
for i in range(1):
    f= open(all_files[i],"w+")
    annotationLines = open( "data/darknet53/"+all_files[i], "r")
    for line in annotationLines:
        count= count + 1
        line = line.split(" ")
        bbox.append(line[1])
        bbox.append(line[2])
        bbox.append(line[3])
        bbox.append(line[4].rstrip("\n"))
    print(bbox)


# In[19]:


import matplotlib.pyplot as pyplot
from matplotlib.patches import Rectangle
import matplotlib as mpl

# draw all bounding boxes
def draw_boxes(filename):
    dpi = mpl.rcParams['figure.dpi']
    im_data = pyplot.imread(filename)
    heightI, widthI, depth = im_data.shape
    figsize = widthI / float(dpi), heightI / float(dpi)
    fig = pyplot.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('on')
    data = pyplot.imread(filename)
    pyplot.imshow(data)
    ax = pyplot.gca()
    rect1 = Rectangle((243,338),600,170, fill=False, color='cyan')
    rect2 = Rectangle((318,270),537,110, fill=False, color='cyan')
    rect3 = Rectangle((419,250),389,77, fill=False, color='cyan')
    rect4 = Rectangle((438,188),434,98, fill=False, color='cyan')
    
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.add_patch(rect4)
    
    pyplot.show()


# In[20]:


draw_boxes('./VisDrone/VisDrone2019-DET-train/images/0000008_00889_d_0000039.jpg')


# In[ ]:




