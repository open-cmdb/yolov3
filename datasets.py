import os
import sys
import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils import functions as c_functions

import utils.anchors as l_anchors

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


class Parse:

    def __init__(self, image_dir, anchors, grid_sizes, IMAGE_TARGET_SIZE):
        self.image_dir = image_dir
        self.anchors = anchors
        self.grid_sizes = grid_sizes
        self.IMAGE_TARGET_SIZE = IMAGE_TARGET_SIZE

        self.flat_anchors = [tf.reshape(item, (-1, item.shape[-1])) for item in self.anchors]

        self.last_dim = 6

    def get_image(self, image_name):
        image_real_path = self.image_dir + "/" + image_name + ".jpg"
        image = tf.io.read_file(image_real_path)
        image = tf.image.decode_jpeg(image)
        image = tf.image.resize(image, self.IMAGE_TARGET_SIZE)
        image = tf.cast(image, tf.float32)
        image = image / 127.5 - 1
        return image

    def get_label(self, label):
        label = tf.strings.regex_replace(label, " ", ",")
        label = tf.strings.split(label, ",")
        label = tf.strings.to_number(label)
        label = tf.reshape(label, (-1, 5))

        boxes = label[..., 0:4]
        instance_ids = label[..., 4]

        # one_dim_anchors = tf.concat(self.flat_anchors, 0)
        # shape = one_dim_anchors.shape[:-1] + [6]
        # one_dim_labels = tf.zeros(shape)
        # shape = one_dim_anchors.shape[0:1] + [1]
        # tiled_boxes = tf.tile(boxes, shape)
        # shape = one_dim_anchors.shape[0:1] + boxes.shape
        # tiled_boxes = tf.reshape(tiled_boxes, shape)
        # ious = c_functions.calc_iou(one_dim_anchors, tiled_boxes)

        labels = []
        for item in self.anchors:
            shape = item.shape[:-1] + [self.last_dim]
            layer_label = tf.ones(shape) * -1
            labels.append(layer_label)

        for item in label:
            box = item[..., :4]
            id = item[..., 4]

            box_xy = box[0:2] + (box[2:4] - box[0:2]) / 2

            max_index = 0
            current_max_iou = 0.
            current_arg_max = None
            current_box_xy_index = None
            for index, (layer_anchors, cell_size) in enumerate(zip(self.anchors, self.grid_sizes)):
                box_xy_index = box_xy / (1 / cell_size)
                box_xy_index = tf.cast(box_xy_index, tf.int32)
                box_xy_index = box_xy_index.numpy().tolist()
                anchors = layer_anchors[box_xy_index]
                ious = c_functions.calc_iou(box, anchors)

                arg_max = tf.argmax(ious, output_type=tf.int32)
                max_iou = tf.reduce_max(ious)
                if max_iou > current_max_iou:
                    max_index = index
                    current_max_iou = max_iou
                    current_arg_max = arg_max
                    current_box_xy_index = box_xy_index

            value = tf.concat((box, [1.0, id]), 0)
            indices = tf.concat((current_box_xy_index, [current_arg_max]), 0)
            indices = indices[tf.newaxis, ...]
            labels[max_index] = tf.tensor_scatter_nd_update(labels[max_index], indices, [value])

            # max_index = 0
            # arg_max_list = []
            # current_max_iou = 0.
            # current_arg_max = None
            # for index, layer_anchors in enumerate(self.flat_anchors):
            #     ious = c_functions.calc_iou(box, layer_anchors)
            #     max_iou = tf.reduce_max(ious)
            #     arg_max = tf.argmax(ious, output_type=tf.int32)
            #     arg_max_list.append(arg_max)
            #     if max_iou > current_max_iou:
            #         current_max_iou = max_iou
            #         max_index = index
            #         current_arg_max = arg_max
            # # arg_max = arg_max_list[max_index]
            # value = tf.concat((box, [1.0, id]), 0)
            # indices = current_arg_max[tf.newaxis, tf.newaxis, ...]
            # labels[max_index] = tf.tensor_scatter_nd_update(labels[max_index], indices, [value])

        return labels

        ret_labels = []
        for layer_labels, layer_anchors in zip(labels, self.anchors):
            target_shape = layer_anchors.shape[:-1] + [self.last_dim]
            ret_layer_labels = tf.reshape(layer_labels, target_shape)
            ret_labels.append(ret_layer_labels)
        return ret_labels

    def __call__(self, item):

        splited_list = tf.strings.split(item, " ", 1)
        image_name = splited_list[0]
        image = self.get_image(image_name)

        label = splited_list[1]
        label1, label2, label3 = tf.py_function(self.get_label, [label], [tf.float32, tf.float32, tf.float32])
        # label1.set_shape(self.flat_anchors[0].shape[:1]+[6])
        # return image, label1, label2
        # label1.set_shape(self.anchors[0].shape[:-1] + [6])
        # label2.set_shape(self.anchors[1].shape[:-1] + [6])
        # label3.set_shape(self.anchors[2].shape[:-1] + [6])
        # label = self.get_label(label)

        label1 = tf.reshape(label1, [-1, self.last_dim])
        label2 = tf.reshape(label2, [-1, self.last_dim])
        label3 = tf.reshape(label3, [-1, self.last_dim])

        label = tf.concat((label1, label2, label3), 0)

        return image, (label[..., 0:4], label[..., 4:5], label[..., 5:6], label)

