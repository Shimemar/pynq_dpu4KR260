#!/usr/bin/env python
# coding: utf-8

# # DPU example: YOLOx-nano
# ----

# In[ ]:





# In[1]:


# ***********************************************************************
# Import Packages
# ***********************************************************************


# In[1]:


from pynq_dpu import DpuOverlay
overlay = DpuOverlay("./mydpu/dpu2.bit")


# In[2]:


import os
import time
import numpy as np
import cv2
import random
import colorsys
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from copy import deepcopy
#from IPython.display import display, Image, clear_output

import copy


# In[3]:


overlay.load_model("yolox_nano_pt.xmodel")


# In[4]:


cap = cv2.VideoCapture(0) # カメラからデバイス情報を取得
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416) # 表示画像の幅を設定
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 416) # 表示画像の高さを設定
cap.set(cv2.CAP_PROP_FPS, 60) # 表示画像のfpsを設定


# In[5]:


# 画像を出力する関数
def imshow(frame):
   # 文字を描画（FPS)
    _, buf = cv2.imencode(".jpg", frame) # エンコード方法にJPEGを指定
    display(Image(data=buf.tobytes()))


# In[6]:


ret , frame = cap.read() 
orig_frame = deepcopy(frame)
imshow(orig_frame)


# In[8]:


# ***********************************************************************
# Use VART APIs
# ***********************************************************************


# In[9]:


dpu = overlay.runner


# In[11]:


inputTensors = dpu.get_input_tensors()


# In[13]:


outputTensors = dpu.get_output_tensors()


# In[15]:


shapeIn = tuple(inputTensors[0].dims)


# In[17]:


shapeOut0 = (tuple(outputTensors[0].dims)) # (1, 52, 52, 85)
shapeOut1 = (tuple(outputTensors[1].dims)) # (1, 26, 26, 85)
shapeOut2 = (tuple(outputTensors[2].dims)) # (1, 13, 13, 85)


# In[19]:


outputSize0 = int(outputTensors[0].get_data_size() / shapeIn[0]) # 229840
outputSize1 = int(outputTensors[1].get_data_size() / shapeIn[0]) # 57460
outputSize2 = int(outputTensors[2].get_data_size() / shapeIn[0]) # 14365


# In[21]:


input_data = [np.empty(shapeIn, dtype=np.float32, order="C")]
output_data = [np.empty(shapeOut0, dtype=np.float32, order="C"), 
               np.empty(shapeOut1, dtype=np.float32, order="C"),
               np.empty(shapeOut2, dtype=np.float32, order="C")]
image = input_data[0]


# In[ ]:





# In[22]:


def preprocess(image, input_size, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_image = np.ones(
            (input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_image = np.ones(input_size, dtype=np.uint8) * 114

    ratio = min(input_size[0] / image.shape[0],
                input_size[1] / image.shape[1])
    resized_image = cv2.resize(
        image,
        (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    resized_image = resized_image.astype(np.uint8)

    padded_image[:int(image.shape[0] * ratio), :int(image.shape[1] *
                                                    ratio)] = resized_image
    #padded_image = padded_image.transpose(swap)

    padded_image = np.ascontiguousarray(padded_image, dtype=np.float32)
    return padded_image, ratio


# In[23]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


# In[24]:


def postprocess(
    outputs,
    img_size,
    ratio,
    nms_th,
    nms_score_th,
    max_width,
    max_height,
    p6=False,
):
    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    predictions = outputs[0]
    boxes = predictions[:, :4]
    scores = sigmoid(predictions[:, 4:5]) * softmax(predictions[:, 5:])
    #scores = predictions[:, 4:5] * predictions[:, 5:]
    
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    boxes_xyxy /= ratio

    dets = multiclass_nms(
        boxes_xyxy,
        scores,
        nms_thr=nms_th,
        score_thr=nms_score_th,
    )

    bboxes, scores, class_ids = [], [], []
    if dets is not None:
        bboxes, scores, class_ids = dets[:, :4], dets[:, 4], dets[:, 5]
        for bbox in bboxes:
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(bbox[2], max_width)
            bbox[3] = min(bbox[3], max_height)

    return bboxes, scores, class_ids


# In[25]:


def nms(boxes, scores, nms_thr):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


# In[26]:


def multiclass_nms(
    boxes,
    scores,
    nms_thr,
    score_thr,
    class_agnostic=True,
):
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware

    return nms_method(boxes, scores, nms_thr, score_thr)


# In[27]:


def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    final_dets = []
    num_classes = scores.shape[1]

    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr

        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = self._nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [
                        valid_boxes[keep], valid_scores[keep, None],
                        cls_inds
                    ],
                    1,
                )
                final_dets.append(dets)

    if len(final_dets) == 0:
        return None

    return np.concatenate(final_dets, 0)


# In[28]:


def multiclass_nms_class_agnostic(boxes, scores, nms_thr,
                                    score_thr):
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr

    if valid_score_mask.sum() == 0:
        return None

    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)

    dets = None
    if keep:
        dets = np.concatenate([
            valid_boxes[keep],
            valid_scores[keep, None],
            valid_cls_inds[keep, None],
        ], 1)

    return dets


# In[29]:


'''Get model classification information'''	
def get_class(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names
    
classes_path = "img/coco2017_classes.txt"
class_names = get_class(classes_path)


# In[30]:


'''Draw detection frame'''
def draw_bbox(image, bboxes, classes):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
    return image


# In[31]:


# ***********************************************************************
# Main Program
# ***********************************************************************


# In[39]:


def run(image_index, display=False):

    input_shape=(416, 416)
    class_score_th=0.3
    nms_th=0.45
    nms_score_th=0.1
    start_time = time.time()
    
    #input_image = cv2.imread(os.path.join(image_folder, original_images[image_index]))
    ret , input_image = cap.read() # デバイス情報から読み込み情報と画像を取得
    
    start_time = time.time()
    # Pre-processing
    pre_process_start = time.time()
    #temp_image = copy.deepcopy(input_image)
    image_height, image_width = input_image.shape[0], input_image.shape[1]
    image_size = input_image.shape[:2]
    image_data, ratio = preprocess(input_image, input_shape)
    pre_process_end = time.time()
    
    # Fetch data to DPU and trigger it
    dpu_start = time.time()
    image[0,...] = image_data.reshape(shapeIn[1:])    
    #output_folder = "img/"
    #result_path = os.path.join(output_folder, f'preprocess.jpg')
    #cv2.imwrite(result_path, image[0])
    job_id = dpu.execute_async(input_data, output_data)
    dpu.wait(job_id)
    dpu_end = time.time()
    
    # postprocess
    decode_start = time.time()
    outputs = np.concatenate([output.reshape(1, -1, output.shape[-1]) for output in output_data], axis=1)
    bboxes, scores, class_ids = postprocess(
        outputs,
        input_shape,
        ratio,
        nms_th,
        nms_score_th,
        image_width,
        image_height,
    )
    decode_end = time.time()
    
    end_time = time.time()
    
    # draw_bbox
    draw_start = time.time()
    if display:
        bboxes_with_scores_and_classes = []
        for i in range(len(bboxes)):
            bbox = bboxes[i].tolist() + [scores[i], class_ids[i]]
            bboxes_with_scores_and_classes.append(bbox)
        bboxes_with_scores_and_classes = np.array(bboxes_with_scores_and_classes)
        display = draw_bbox(input_image, bboxes_with_scores_and_classes, class_names)
        output_folder = "img/"
        result_path = os.path.join(output_folder, f'result.jpg')
        cv2.imwrite(result_path, display)
    draw_end = time.time()
       
#    clear_output(wait=True)
    print("Details of detected objects: {}".format(class_ids))
    print("Pre-processing time: {:.4f} seconds".format(pre_process_end - pre_process_start))
    print("DPU execution time: {:.4f} seconds".format(dpu_end - dpu_start))
    print("Post-process time: {:.4f} seconds".format(decode_end - decode_start))
    #print("Draw boxes time: {:.4f} seconds".format(draw_end - draw_start))
    print("Total run time: {:.4f} seconds".format(end_time - start_time))
    print("Performance: {} FPS".format(1/(end_time - start_time)))
#    imshow(display)
    
    return bboxes, scores, class_ids


# In[42]:


id = run(0, display=True)
print(id)


# In[44]:


total_images = 600

for i in range(1,total_images):
    ret = run(0, display=True)

#    if (ret == 14).any():
#        gpio_out.write(0x7,mask)
#    else:
#        gpio_out.write(0x0,mask)



# In[35]:


del overlay
del dpu


# In[ ]:




