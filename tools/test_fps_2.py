import os
import sys
import time
import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


import utils.anchors as l_anchors
import datasets as l_datasets
import losses as l_losses
import metrics as l_metrics
import models as l_models
import config as l_config

gpus = tf.config.experimental.list_physical_devices("gpu")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

anchor_sizes = tf.convert_to_tensor(l_config.anchor_sizes, tf.float32)
anchor_sizes = anchor_sizes / l_config.image_target_size[0]
anchor_instance = l_anchors.Anchor(anchor_sizes, l_config.grid_sizes, l_config.image_target_size)
anchors = anchor_instance.get_anchors()

train_parse = l_datasets.Parse(l_config.train_image_dir, anchors, l_config.grid_sizes, l_config.image_target_size)
val_parse = l_datasets.Parse(l_config.val_image_dir, anchors, l_config.grid_sizes, l_config.image_target_size)

model = l_models.YoloV3(l_config.filters, anchors, l_config.grid_sizes, l_config.class_num)

last_time = time.time()
count = 0
images = tf.ones([1] + list(l_config.image_target_size) + [3])


@tf.function
def pre(images):
    return model(images, training=False)


while True:
    now = time.time()
    if now - last_time > 1:
        print(now - last_time)
        print("fps: ", count)
        last_time = now
        count = 0
    pred = pre(images)
    count += 1
