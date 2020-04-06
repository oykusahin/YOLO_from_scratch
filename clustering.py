import numpy as np
import os
from PIL import Image

def iou(box, clusters):
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou_ = intersection / (box_area + cluster_area - intersection)
    return iou_

def kmeans(boxes, k, dist=np.median):
    rows = boxes.shape[0]
    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))
    np.random.seed()
    clusters = boxes[np.random.choice(rows, k, replace=False)]
    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)
        nearest_clusters = np.argmin(distances, axis=1)
        if (last_clusters == nearest_clusters).all():
            break
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
        last_clusters = nearest_clusters
    return clusters

annotations_dir = "VisDrone2019-DET-train/annotations"
all_files = os.listdir(annotations_dir)
boxes= {}
for i in range(len(all_files)):
    f= open(annotations_dir +"/" + all_files[i],"r")
    annotationLines = open( "VisDrone2019-DET-train/annotations/"+all_files[i], "r")
    im_dir ="VisDrone2019-DET-train/images/"+all_files[i].rstrip(".txt")+".jpg"  
    im = Image.open(im_dir,"r") 
    width, height = im.size
    box = []
    for line in annotationLines:
        line = line.rstrip('\n')
        line = line.split(",")
        
        w=float(float(line[2])/width)
        h=float(float(line[3])/height)
        box = np.append(box , w)
        box = np.append(box , h)
        
    boxes[str(i)] = box

for i in range(len(boxes)):
    box = np.reshape(boxes[str(i)],(-1, 2))
    boxes[str(i)] = box
    clusters = kmeans(boxes[str(i)], 2)
    print(clusters)
        