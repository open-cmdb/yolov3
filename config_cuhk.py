import os
import sys
import datetime

board_log_dir = "/opt/tensorflow/tensorboard/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
save_weight_file = f"/opt/tensorflow/checkpoints/{__file__.replace('/', '-').replace('.', '-').lstrip('-')}/weights"

anchor_sizes = [[[38.69415527948574, 13.210888737237196], [55.44060121398979, 18.99360437572678],
                 [73.62774965739038, 25.915084546974132]],
                [[94.10204376921686, 32.73846713558089], [117.8882731958763, 41.16864919354839],
                 [143.48247422680413, 52.10710569347406]],
                [[172.3619967793881, 61.503994412164126], [222.5129030739538, 64.21574079919196],
                 [329.8654970760234, 96.55350877192984]]]

image_target_size = (416, 416)
grid_sizes = [52, 26, 13]
layer_num = len(grid_sizes)
cell_anchor_num = 3
train_label_file = "/opt/datasets/cuhk/VOCdevkit/VOC2012/train_labels.txt"
train_image_dir = "/opt/datasets/cuhk/VOCdevkit/VOC2012/JPEGImages"
val_label_file = "/opt/datasets/cuhk/VOCdevkit/VOC2012/val_labels.txt"
val_image_dir = "/opt/datasets/cuhk/VOCdevkit/VOC2012/JPEGImages"
class_num = 1
batch_size = 4

metric_iou_threshold = 0.5
metric_score_threshold = 0.8

nms_iou_threshold = 0.1
nms_score_threshold = metric_score_threshold
nms_max_output = 1000

cate_names = ["person"]