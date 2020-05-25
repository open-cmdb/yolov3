import os
import sys

import tensorflow as tf

import matplotlib.pyplot as plt


current_path = os.getcwd()
project_root_path = current_path.split("mechine_learning")[0]
utils_path = os.path.join(project_root_path, "utils")
sys.path.insert(0, utils_path)

from utils import functions as c_functions


class LocationLoss(tf.keras.losses.Loss):

    def __init__(self, anchors):
        super().__init__()

        flat_anchors = [tf.reshape(item, (-1, 4)) for item in anchors]
        self.flat_anchors = tf.concat(flat_anchors, 0)

    def draw(self, y_true, y_pred):
        images = tf.zeros((1, 416, 416, 3))

        is_object = y_true[..., 0] != -1
        y_true = tf.boolean_mask(y_true[0], is_object[0])
        y_pred = tf.boolean_mask(y_pred[0], is_object[0])
        anchors = tf.boolean_mask(self.flat_anchors, is_object[0])

        images = tf.image.draw_bounding_boxes(images, [y_true[:5]], [[0, 255, 0]])
        images = tf.image.draw_bounding_boxes(images, [y_pred[:5]], [[255, 0, 0]])
        images = tf.image.draw_bounding_boxes(images, [anchors[:5]], [[0, 0, 255]])

        # images = tf.image.draw_bounding_boxes(images, [y_pred[0][1000:1004]], [[0, 255, 0]])
        # images = tf.image.draw_bounding_boxes(images, [self.flat_anchors[1000:1004]], [[0, 0, 255]])

        image = tf.cast(images[0], tf.int32).numpy()
        plt.imshow(image)
        plt.show()
        print()

    def call(self, y_true, y_pred):

        # self.draw(y_true, y_pred)

        is_object = y_true[..., 0] != -1
        # if tf.reduce_all(is_object == False):
        #     return tf.concat(0.0, tf.float32)

        y_true = tf.where(y_true == -1, 0.0, y_true)

        giou = c_functions.calc_giou(y_true, y_pred)
        loss = 1 - giou
        loss = tf.boolean_mask(loss, is_object)
        loss = tf.concat((loss, [0.0]), 0)
        loss = tf.reduce_mean(loss)
        return loss


class ConfidenceLoss(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        # is_object = y_true[..., 0] != -1.0
        # is_object = tf.cast(is_object, tf.float32)

        y_true = tf.where(y_true == -1, 0.0, y_true)
        is_not_object = tf.cast(y_true[..., 0] == 0.0, tf.float32)

        positive_num = tf.reduce_sum(y_true)
        negative_num = tf.reduce_sum(is_not_object)

        positive_num = positive_num + 1
        negative_num = negative_num + 1

        weight = negative_num / positive_num

        loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        positive_loss = loss * y_true[..., 0]
        negative_loss = loss * is_not_object

        positive_loss = positive_loss * weight

        loss = positive_loss + negative_loss

        loss = tf.reduce_mean(loss)
        return loss


class CategoricalLoss(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        is_object = y_true[..., 0] != -1
        # if tf.reduce_all(is_object == False):
        #     return tf.concat(0.0, tf.float32)

        y_true = tf.where(y_true == -1, 0.0, y_true)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        loss = tf.boolean_mask(loss, is_object)
        loss = tf.concat((loss, [0.0]), 0)
        loss = tf.reduce_mean(loss)
        return loss


class AllLoss(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        return tf.constant(0.0, tf.float32)
