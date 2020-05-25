import os
import sys
import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


class Anchor:

    def __init__(self, anchor_sizes, cell_sizes, image_target_size):
        if len(anchor_sizes) != len(cell_sizes):
            raise Exception("len(anchor_sizes) != len(cell_sizes)")
        self.anchor_sizes = anchor_sizes
        self.cell_sizes = cell_sizes
        self.image_target_size = image_target_size

    def get_single_layer_anchors(self, layer_anchor_sizes, cell_size):

        indices = tf.linspace(0.0, 1.0 - 1.0 / cell_size, cell_size)
        xx, yy = tf.meshgrid(indices, indices)
        grid = tf.stack((yy, xx), -1)
        grid_center = grid + 1.0 / cell_size / 2.0
        anchors = []
        for anchor_size in layer_anchor_sizes:
            # anchor_size = layer_anchor_sizes[layer_anchor_sizes * index:layer_anchor_sizes * (index + 1)]
            anchor_size = tf.convert_to_tensor(anchor_size)
            anchor_size_half = anchor_size / 2
            pt1 = grid_center - anchor_size_half
            pt2 = grid_center + anchor_size_half
            boxes = tf.concat((pt1, pt2), -1)
            anchors.append(boxes)
        anchors = tf.stack(anchors, -2)
        # anchors = tf.clip_by_value(anchors, 0, 1)
        return anchors

    def get_anchors(self):
        all_anchors = []
        for layer_anchor_sizes, cell_size in zip(self.anchor_sizes, self.cell_sizes):
            anchors = self.get_single_layer_anchors(layer_anchor_sizes, cell_size)
            all_anchors.append(anchors)
        return all_anchors

    # def get_anchors(self):
    #     all_anchors = []
    #     for index, cell_size in enumerate(self.cell_sizes):
    #         if len(self.anchor_sizes) % len(self.cell_sizes) != 0:
    #             raise Exception("len(self.anchor_sizes) % len(self.cell_sizes) != 0")
    #         layer_anchor_len = len(self.anchor_sizes) // len(self.cell_sizes)
    #         layer_anchor_sizes = self.anchor_sizes[index * layer_anchor_len:(index + 1) * layer_anchor_len]
    #         anchors = self.get_single_layer_anchors(layer_anchor_sizes, cell_size)
    #         all_anchors.append(anchors)
    #     return all_anchors


if __name__ == '__main__':
    anchor_sizes = [[[42.73488708, 32.45970021],
                     [110.11641671, 63.71981952],
                     [206.75434957, 91.45455189]],
                    [[118.08332025, 168.44630685],
                     [321.75786016, 132.5452512],
                     [225.20104213, 218.92525501]],
                    [[201.29191875, 359.45219147],
                     [356.20329286, 256.36012672],
                     [369.89560187, 380.75450991]], ]
    image_target_size = (416, 416)
    cell_sizes = [32, 16, 8]

    anchor_sizes = tf.convert_to_tensor(anchor_sizes, tf.float32)
    anchor_sizes = anchor_sizes / image_target_size[0]
    anchor_instance = Anchor(anchor_sizes, cell_sizes, image_target_size)
    anchors = anchor_instance.get_anchors()

    image = tf.zeros(list(image_target_size)+[3])
    anchor_layer_index = 2
    boxes = anchors[anchor_layer_index][5, 0, ...]
    images = tf.image.draw_bounding_boxes([image], [boxes], [[255, 0, 0]])
    image = images[0].numpy().astype(np.int32)
    plt.imshow(image)
    plt.show()
    print()
