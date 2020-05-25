import os
import sys
import json
import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import config as l_config


anchor_num = l_config.layer_num * l_config.cell_anchor_num
label_files = [l_config.train_label_file]
if l_config.val_label_file:
    label_files.append(l_config.val_label_file)


all_boxes = []

for item in label_files:
    with open(item) as f:
        label_content = f.read()

    label_content = label_content.strip("\n")
    lines = label_content.split("\n")

    for item in lines:
        splits = item.split(" ")
        name = splits[0]
        boxes = [[float(i) for i in item.split(",")[:4]] for item in splits[1:]]
        all_boxes.extend(boxes)

all_boxes = np.array(all_boxes)
boxes_size = all_boxes[..., 2:4] - all_boxes[..., 0:2]

kmeans = KMeans(anchor_num)
pred = kmeans.fit_predict(boxes_size)

anchors = []

for i in range(anchor_num):
    boxes = boxes_size[pred == i]
    mean_height = boxes[..., 0].mean()
    mean_width = boxes[..., 1].mean()
    anchors.append([mean_height, mean_width])

anchors = np.array(anchors)
anchors_area = anchors[..., 0] * anchors[..., 1]
anchors_area_sort = np.sort(anchors_area)
arg_sort = np.argsort(anchors_area)
anchors = anchors[arg_sort]
anchors = anchors * np.array(l_config.image_target_size)
shape = [l_config.layer_num, l_config.cell_anchor_num, 2]
anchors = np.reshape(anchors, shape)
anchors = anchors.tolist()
print(anchors)
