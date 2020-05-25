
import os
import sys
import datetime

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import datasets as l_datasets
import config as l_config
import utils.anchors as l_anchors


gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

anchor_sizes = tf.convert_to_tensor(l_config.anchor_sizes, tf.float32)
anchor_sizes = anchor_sizes / l_config.image_target_size[0]

anchor_instance = l_anchors.Anchor(anchor_sizes, l_config.grid_sizes, l_config.image_target_size)
anchors = anchor_instance.get_anchors()

parse = l_datasets.Parse(l_config.train_image_dir, anchors, l_config.grid_sizes, l_config.image_target_size)


train_ds = tf.data.TextLineDataset(l_config.train_label_file)
train_ds = train_ds.map(parse)

flat_anchors = [tf.reshape(item, (-1, 4)) for item in anchors]
flat_anchors = tf.concat(flat_anchors, 0)

for index, (image, label) in enumerate(train_ds.take(3)):
    image = (image + 1.0) * 127.5
    images = [image]
    layer_conf = label[1]
    mask = layer_conf[..., 0] == 1
    mask_boxes = tf.boolean_mask(label[0], mask)
    mask_anchors = tf.boolean_mask(flat_anchors, mask)
    mask_cates = tf.boolean_mask(label[2][..., 0], mask)
    cates = tf.boolean_mask(label[2][..., 0], mask)
    images = tf.image.draw_bounding_boxes(images, [mask_boxes], [[0, 255, 0]])
    images = tf.image.draw_bounding_boxes(images, [mask_anchors], [[255, 0, 0]])

    image = images[0].numpy().astype(np.int32)

    cv2_loca = mask_boxes.numpy()[..., :2] * np.array(l_config.image_target_size)
    cv2_loca = cv2_loca.astype(np.int32)
    cv2_loca[..., 0] += 10
    cv2_cate = mask_cates.numpy().astype(np.int32)
    for index, (loca, cate) in enumerate(zip(cv2_loca, cv2_cate)):
        loca = tuple(reversed(loca))
        cate_name = l_config.cate_names[cate]
        image = cv2.putText(image, cate_name, loca, cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.show()
    print()
