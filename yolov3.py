import urllib.request
import os
from zipfile import ZipFile
from PIL import Image
from __future__ import division
from os import getcwd
import xml.etree.ElementTree as ET
import glob, os
from torch.autograd import Variable

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import cv2

path = os.getcwd()
print ('The current working directory is %s' % path)

##Creates the new directory to save the dataset
path = path+'\yolov3\Dataset'
if len(os.listdir(path) ) == 0:
    try:
        os.mkdir(path)
    except OSError:
        print ('Creation of the directory %s failed' % path)
    else:
        print ('Successfully created the directory %s ' % path)
    
    print('Beginning COCO Dataset download.')
    urlImages = 'http://images.cocodataset.org/zips/train2017.zip'
    urllib.request.urlretrieve(urlImages, path)
    dirImages = path+'/train2017.zip'
    urlAnnotations = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    urllib.request.urlretrieve(urlAnnotations, path)
    dirAnnotations = path+'/annotations_trainval2017.zip'
    
    ###Unzipping the Images and Annotations
    with ZipFile(dirImages, 'r') as zipObj:
        zipObj.extractall()
    with ZipFile(dirAnnotations, 'r') as zipObj:
        zipObj.extractall()



image = {}

def convert_image(line):
    file_dir= path+'/train2017/'+line
    im = Image.open(file_dir, "r")
    temp_data = np.asarray(im.getdata())
    pix_val = np.resize(temp_data,(608, 608, 3))
    line = line.replaceFirst('^0+(?!$)', '')
    image[line]= pix_val

dirImages = path+'/train2017'
f = open(dirImages, "r+")
lines= f.readlines()
count=0
for line in lines:
    count+=1
    convert_image(line)


# In[38]:


from keras.utils import to_categorical
annotationDict = {}
def convert_annotation(file_dir):
    with open("instances_train2017.json") as datafile:
        data = json.load(datafile)
    
    for i in range(len(data['annotations'])):
        temp=[]
        categ_raw = data['annotations'][i]['category_id']
        categ.append(categ_raw, '81')
        categ = to_categorical(categ)
        temp.append(temp, '1')
        temp.append((data['annotations'][i]['bbox']))
        temp.append(categ)
    
        n=22743-len(categ_raw)
        tempArray = np.zeros(n*85)
        temp.append(tempArray)
        
        annotationDict[data['annotations'][i]['image_id']] = temp
    return annotationDict

dirAnnotations = path+'/annotations_trainval2017/instances_train2017.json'
tempy_train = convert_annotation(dirAnnotations)


# In[ ]:


X_train=[]
Y_train=[]
for i in range(len(tempy_train)):
    key_at_index = dic.key[i]
    X_train.append(image[key_at_index])
    Y_train.append(tempy_train[key_at_index])

X_train = X_train.reshape(len(tempy_train), 608, 608, 3)
Y_train= Y_train.reshape(len(tempy_train), 22743, 3)


# In[ ]:


def parse_cfg(cfgfile):
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                        
    lines = [x for x in lines if len(x) > 0]               
    lines = [x for x in lines if x[0] != '#']              
    lines = [x.rstrip().lstrip() for x in lines]           

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":               
            if len(block) != 0:         
                blocks.append(block)     
                block = {}               
            block["type"] = line[1:-1].rstrip()    
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks


# In[ ]:


def create_modules(blocks):
    net_info = blocks[0]     #Captures the information about the input and pre-processing    
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []
    
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        
        if (x["type"] == "convolutional"):
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)
        

        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module("upsample_{}".format(index), upsample)
                
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
     
            start = int(x["layers"][0])
            try:
                end = int(x["layers"][1])
            except:
                end = 0

            if start > 0: 
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]
    

        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
            
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
    
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
    
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)
                              
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        
    return (net_info, module_list)


# In[ ]:


def predict_transform(prediction, inp_dim, anchors, num_classes):

    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)
    
    x_offset = torch.DoubleTensor(a).view(-1,1)

    y_offset = torch.DoubleTensor(b).view(-1,1)
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    prediction[:,:,:2] += x_y_offset

    anchors = torch.DoubleTensor(anchors)

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride
    
    return prediction


# In[ ]:


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
     
    #taken from github
    def load_weights(self, weightfile):
        fp = open(weightfile, "rb")
  
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                conv = model[0]                
                if (batch_normalize):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

    def forward(self, x, labels):
        modules = self.blocks[1:]
        outputs = {}   #We cache the outputs for the route layer
        loss=0
        print(labels.shape)
        write = 0
        for i, module in enumerate(modules):        
            module_type = (module["type"])
            
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
    
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
    
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
    
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
    
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
    
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                    print("Route: " + str(x.shape))
                
    
            elif  module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]
    
            elif module_type == 'yolo':        
                anchors = self.module_list[i][0].anchors
                inp_dim = int (self.net_info["height"])
                num_classes = 20
                x = x.data
                layer_loss= 0
                x = predict_transform(x, inp_dim, anchors, num_classes)
                if not write:              #if no collector has been intialised. 
                    detections = x
                    write = 1
                else:       
                    detections = torch.cat((detections, x), 1)
                    yolo_loss(detections, labels)
                    loss += layer_loss
            outputs[i] = x
        
        return detections, loss


# In[ ]:


def bbox_iou(box1, box2):
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou


# In[ ]:


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


# In[ ]:


def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    #For each of the bounding box having a objectness score below a threshold, 
    #we set the values of it's every attribute to zero.

    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    
    #transform the (center x, center y, height, width) attributes of our boxes, 
    #          to  (top-left corner x, top-left corner y, right-bottom corner x, right-bottom corner y)
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]
    #
    
    batch_size = prediction.size(0)
    write = False
    #NMS has to be done for one image at once.
    #we cannot vectorise the operations involved
    for ind in range(batch_size):
        #remove the 80 class scores from each row
        #add the index of the class having the maximum values and class score.
        image_pred = prediction[ind]      
      
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.double().unsqueeze(1)
        max_conf_score = max_conf_score.double().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        
        #get rid of bounding box rows having a object confidence less than the threshold
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        # handle situations where we get no detections.
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        
        if image_pred_.shape[0] == 0:
            continue       

        #Since there can be multiple true detections of the same class 
        #to get classes present in any given image.
        img_classes = unique(image_pred_[:,-1])  
        #extract the detections of a particular class
        for cls in img_classes:
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)  
            #NMS
            for i in range(idx):
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
            
                except IndexError:
                    break
            
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       

                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            #D x 8. Here D is the true detections in all of images, each represented by a row. 
            #Each detections has 8 attributes, namely, index of the image in the batch to which the detection belongs to, 
            #4 corner coordinates, objectness score, the score of class with maximum confidence, and the index of that class.
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))

    try:
        return output
    except:
        return 0


# In[ ]:


import torch.nn as nn
def yolo_loss(labels, predictions):
    loss_x = mse_loss(labels[:,:1], predictions[:,:1])
    loss_y = mse_loss(labels[:,1:2], predictions[:,1:2])
    loss_w = mse_loss(labels[:,2:3], predictions[:,2:3])
    loss_h = mse_loss(labels[:,3:4], predictions[:,3:4])
    loss_conf_obj = bce_loss(labels[:,4:5], predictions[:,4:5])
    loss_conf = obj_scale * loss_conf_obj
    total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf
    return total_loss


# In[ ]:


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# In[ ]:


##Load config file
urlConfig = 'https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg'
urllib.request.urlretrieve(urlImages, path)

filename= path+'yolov3.cfg'
net = Darknet(filename)
for epoch in range(2):
    net.train()
    running_loss = 0.0
    for i in enumerate(X_trainTF, 0):
        inputs=X_train
        labels=Y_train

        optimizer.zero_grad()

        loss, outputs = net.forward(inputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000)) 
        running_loss = 0.0

print('Finished Training')

