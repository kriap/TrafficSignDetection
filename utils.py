import numpy as np
import imageio as io
import matplotlib.pyplot as plt
import json
import os, sys
import numpy as np
import pandas as pd
from PIL import Image
import glob
from skimage.transform import resize
from skimage.color import rgb2gray
import matplotlib.patches as patches
import cv2
import torch
from torch.autograd import Variable
from torch.utils.data.dataloader import default_collate

#Red Circle signs
prohibitory = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16] 


#Non-Red Signs 

mandatory = [33, 34, 35, 36, 37, 38, 39, 40] #(circular, blue ground)

def prepare_data(data_directory,labelpath):
    
    """
    Description:
        
    Arguments:
        labelpath : Text file containing ground truth informations. 'txt'

    Returns:
        data (dict) : dictionary containing ground truth label information of all red round signs
        X_train (list): a list of tuples (imagepaths, box_coordinates) , where imagepaths is path to red/blue round signs in the dataset
    """
    
    X = []

    with open(labelpath, 'r') as f:
        for line in f:
            [image_file, x1,y1,x2,y2,class_id] = line.split(';')
            
            key = image_file.split('.')[0]
            
            image_directory = data_directory + image_file 

            
            if int(class_id) in prohibitory:
                
                
                box = (int(x1),int(y1),int(x2),int(y2), int(1))
                                

                
            elif int(class_id) in mandatory:
                                
                box = (int(x1),int(y1),int(x2),int(y2), int(0))

            else:
                
                continue
            
            
            X.append((image_directory, box))
            
    return X
            

def load_annotations(box_coords, scales,size):
    
    """
    Description:
        Loads the bounding box coordinates and returns new coordinates on resized images.

    Arguments:
        box_coords (tuple) : tuple containing the coordinates of left corner and right bottom corner of gt detections
        scales (tuple): resize scaling factor

    Returns:
        annotations (tuple)
    """
        
    (y_scale, x_scale) = scales
    
    annotations = []
    (x1,y1,x2,y2,class_id) = box_coords
    
    x1 = int(float(x1*x_scale))
    y1 = int(float(y1*y_scale))
    x2 = int(float(x2*x_scale))
    y2 = int(float(y2*y_scale))
    
    #dw = 1./size[1]
    #dh = 1./size[0]
    
    #x = (x1 + x2)/2.0
    #y = (y1 + y2)/2.0
    
    #w = x2 - x1
    #h = y2 - y1
    
    #x = x*dw
    #w = w*dw
    #y = y*dh
    #h = h*dh

    
    annotations = (x1,y1,x2,y2,class_id)
    annotations = np.array(annotations)
    return annotations

def data_iterator(fs, seed=0, batch_size=4, num_fs=0, num_batch=0,
                  size=(32, 32), x_offset=0, y_offset=0):
    """
    Description:
        Iterate through a set of images and return them

    Arguments:
        fs (list): a list of tuples (imagepaths, box_coordinates) , where imagepath is the path to an image file.
        seed (int): seed for the random number generator
        batch_size (int): number of images to return in every batch

    Returns:
        images (np.ndarray), bouding_box (np.ndarray)
    """
    GRAYSCALE = True
    RESIZE = True
    
    np.random.seed(seed)

    height, width = size

    fs_a = fs[:num_fs]
    fs = fs[num_fs:]
    num = batch_size - num_batch
    num_itr = int(len(fs)/num)
    num_itr_a = int(len(fs_a)/num_batch) if num_batch != 0 else 1
    while True:
        idxs = list(range(0, len(fs)))
        idxs_a = list(range(0, len(fs_a)))
        np.random.shuffle(idxs)
        np.random.shuffle(idxs_a)

        for batch_idx in range(0, num_itr):
            current_indices = idxs[batch_idx*num:batch_idx*num + num]
            batch_idx = batch_idx % num_itr_a
            current_indices_a = idxs_a[batch_idx*num_batch:batch_idx*num_batch + num_batch]

            images_batch = []
            labels_batch = []
            
            for j in range(batch_size):
                if j < num:
                    i = current_indices[j]
                    imagepath, box_coords = fs[i]
                else:
                    i = current_indices_a[j - num]
                    imagepath, box_coords = fs_a[i]

                im = io.imread(imagepath)
                original_x , original_y, channel = np.shape(im)

                
                #RGB to GRAY
                if GRAYSCALE:
                    im = rgb2gray(im)
                    im = np.stack((im,)*3, axis=-1)
                
                    

                #Resize images
                if RESIZE:       
                    im = resize(im,size)
                                        
                    scale_x = (height/original_x)
                    scale_y = (width/original_y)
                    
                else:
                    
                    scale_x = 1
                    scale_y = 1
                    
               
                #Normalization 
                im = im[y_offset:, x_offset:]
                
                if im.dtype == np.uint8:
                    im = im.astype(np.float64) / 255.0
                
                
                encoded = load_annotations(box_coords,(scale_x,scale_y),size)
                images_batch.append(im)
                labels_batch.append(encoded)
                
            images_batch = np.array(images_batch)
            labels_batch = np.array(labels_batch)

            yield images_batch, labels_batch

def draw_rectangle(image, box):
    
    """
    Description:
       Draws the bounding box on resized images.
       
    Arguments:
        image (np.ndarray): numpy array of the image.
        box_coords (tuple): contains the box coordinates of bounding box on resized image.

    Returns:
        None
    """
    
    fig,ax = plt.subplots(1)
  
    (x1,y1,x2,y2,classid) = box
    print(box)
    
    img_h, img_w, _ = np.shape(image)
    
    #img_h, img_w = np.shape(image)
    
    #x1, y1 = ((x + witdth)/2)*img_width, ((y + height)/2)*img_height
    # x2, y2 = ((x - witdth)/2)*img_width, ((y - height)/2)*img_height
    #x1, y1 = int((box[0] + box[2]/2)*img_w), int((box[1] + box[3]/2)*img_h)
    #x2, y2 = int((box[0] - box[2]/2)*img_w), int((box[1] - box[3]/2)*img_h)

    w = x2-x1
    h = y2-y1
    
    # Display the image
    ax.imshow(image)
    
    rect = patches.Rectangle(xy=(x1,y1), height=h, width=w,linewidth=1,edgecolor='r',facecolor='none')
    
    ax.add_patch(rect)


def post_processing(output, image_size, gt_classes, anchors, conf_threshold, nms_threshold):
    
    """
    Reference:
       https://eavise.gitlab.io/lightnet/api/data.html
 
    Description:
       Returns the final boxes after post_processing of network output on testing set.
       
    Arguments:
        output : output from the trained network.
        anchors (list) – 2D list representing anchor boxes
        conf_threshold (Number [0-1]) – Confidence threshold to filter detections
        nms_threshold (Number [0-1])  - Non Maximum supression threshold to prediction boxes
    Returns:
        [x_center, y_center, width, height, confidence, class_id] for every bounding box
    """
    
    
    num_anchors = len(anchors)
    anchors = torch.Tensor(anchors)
    if isinstance(output, Variable):
        output = output.data

    if output.dim() == 3:
        output.unsqueeze_(0)

    batch = output.size(0)
    h = output.size(2)
    w = output.size(3)

    # Compute xc,yc, w,h, box_score on Tensor
    lin_x = torch.linspace(0, w - 1, w).repeat(h, 1).view(h * w)
    lin_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().contiguous().view(h * w)
    anchor_w = anchors[:, 0].contiguous().view(1, num_anchors, 1)
    anchor_h = anchors[:, 1].contiguous().view(1, num_anchors, 1)

    lin_x = lin_x.type(torch.DoubleTensor)
    lin_y = lin_y.type(torch.DoubleTensor)
    anchor_w = anchor_w.type(torch.DoubleTensor)
    anchor_h = anchor_w.type(torch.DoubleTensor)

    output = output.view(batch, num_anchors, -1, h * w)
    output[:, :, 0, :].sigmoid_().add_(lin_x).div_(w)
    output[:, :, 1, :].sigmoid_().add_(lin_y).div_(h)
    output[:, :, 2, :].exp_().mul_(anchor_w).div_(w)
    output[:, :, 3, :].exp_().mul_(anchor_h).div_(h)
    output[:, :, 4, :].sigmoid_()

    with torch.no_grad():
        cls_scores = torch.nn.functional.softmax(output[:, :, 5:, :], 2)
    cls_max, cls_max_idx = torch.max(cls_scores, 2)
    cls_max_idx = cls_max_idx.float()
    #cls_max.mul_(output[:, :, 4, :])

    print(cls_max)
    score_thresh = cls_max > conf_threshold
    
    #print(score_thresh)
    score_thresh_flat = score_thresh.view(-1)

    if score_thresh.sum() == 0:
        predicted_boxes = []
        for i in range(batch):
            predicted_boxes.append(torch.Tensor([]))
    else:
        coords = output.transpose(2, 3)[..., 0:4]
        coords = coords[score_thresh[..., None].expand_as(coords)].view(-1, 4)
        scores = cls_max[score_thresh]
        idx = cls_max_idx[score_thresh]
        coords = coords.type(torch.DoubleTensor)
        scores = scores.type(torch.DoubleTensor)
        idx = idx.type(torch.DoubleTensor)
        detections = torch.cat([coords, scores[:, None], idx[:, None]], dim=1)

        max_det_per_batch = num_anchors * h * w
        slices = [slice(max_det_per_batch * i, max_det_per_batch * (i + 1)) for i in range(batch)]
        det_per_batch = torch.IntTensor([score_thresh_flat[s].int().sum() for s in slices])
        split_idx = torch.cumsum(det_per_batch, dim=0)

        # Group detections per image of batch
        predicted_boxes = []
        start = 0
        for end in split_idx:
            predicted_boxes.append(detections[start: end])
            start = end

    selected_boxes = []
    for boxes in predicted_boxes:
        if boxes.numel() == 0:
            return boxes

        a = boxes[:, :2]
        b = boxes[:, 2:4]
        bboxes = torch.cat([a - b / 2, a + b / 2], 1)
        scores = boxes[:, 4]

        # Sort coordinates by descending score
        scores, order = scores.sort(0, descending=True)
        x1, y1, x2, y2 = bboxes[order].split(1, 1)

        # Compute dx and dy between each pair of boxes (these mat contain every pair twice...)
        dx = (x2.min(x2.t()) - x1.max(x1.t())).clamp(min=0)
        dy = (y2.min(y2.t()) - y1.max(y1.t())).clamp(min=0)

        # Compute iou
        intersections = dx * dy
        areas = (x2 - x1) * (y2 - y1)
        unions = (areas + areas.t()) - intersections
        ious = intersections / unions

        # Filter based on iou (and class)
        conflicting = (ious > nms_threshold).triu(1)

        keep = conflicting.sum(0).byte()
        keep = keep.cpu()
        conflicting = conflicting.cpu()

        keep_len = len(keep) - 1
        for i in range(1, keep_len):
            if keep[i] > 0:
                keep -= conflicting[i]
        if torch.cuda.is_available():
            keep = keep.cuda()

        keep = (keep == 0)
        selected_boxes.append(boxes[order][keep[:, None].expand_as(boxes)].view(-1, 6).contiguous())

    final_boxes = []
    #print(selected_boxes)
    for boxes in selected_boxes:
        if boxes.dim() == 0:
            final_boxes.append([])
        else:
            #final_boxes.append(boxes)

            boxes[:, 0:3:2] *= image_size
            boxes[:, 0] -= boxes[:, 2] / 2
            boxes[:, 1:4:2] *= image_size
            boxes[:, 1] -= boxes[:, 3] / 2
            
            final_boxes.append([[box[0].item(), box[1].item(), box[2].item(), box[3].item(), box[4].item()] for box in boxes])    
            
    return final_boxes