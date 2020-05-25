import os
import sys

import tensorflow as tf

current_path = os.getcwd()
project_root_path = current_path.split("mechine_learning")[0]
utils_path = os.path.join(project_root_path, "utils")
sys.path.insert(0, utils_path)

from utils import functions as c_functions


class Location(tf.keras.metrics.Metric):

    def __init__(self, threshold=0.45):
        super().__init__()
        self.threshold = threshold
        self.positive_number = self.add_weight("positive_number", initializer="zeros")
        self.all_number = self.add_weight("all_number", initializer="zeros")

    def update_state(self, y_true, y_pred, *args, **kwargs):
        y_true_half = y_true[..., 2:4] / 2
        y_pred_half = y_pred[..., 2:4] / 2
        boxes1_pt1 = y_true[..., 0:2] - y_true_half
        boxes1_pt2 = y_true[..., 0:2] + y_true_half

        boxes2_pt1 = y_pred[..., 0:2] - y_pred_half
        boxes2_pt2 = y_pred[..., 0:2] + y_pred_half

        boxes1 = tf.concat((boxes1_pt1, boxes1_pt2), -1)
        boxes2 = tf.concat((boxes2_pt1, boxes2_pt2), -1)

        is_object = y_true[..., 2] > 0
        is_object = tf.cast(is_object, tf.float32)

        iou = c_functions.calc_iou(boxes1, boxes2)
        iou = iou * is_object

        positive = iou > self.threshold
        positive = tf.cast(positive, tf.float32)
        positive_number = tf.reduce_sum(positive)
        self.positive_number.assign_add(positive_number)

        all_number = tf.reduce_sum(is_object)
        self.all_number.assign_add(all_number)

        return self.result()

    def result(self):
        accuracy_rate = self.positive_number / self.all_number
        return accuracy_rate


class Confidence(tf.keras.metrics.Metric):

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.positive_number = self.add_weight("positive_number", initializer="zeros")
        self.all_number = self.add_weight("all_number", initializer="zeros")

    def update_state(self, y_true, y_pred, *args, **kwargs):
        y_true = tf.where(y_true == -1, 0.0, y_true)
        y_pred = y_pred > self.threshold
        y_pred = tf.cast(y_pred, tf.float32)
        is_equal = y_pred == y_true
        is_equal = tf.cast(is_equal, tf.float32)
        positive_number = tf.reduce_sum(is_equal)
        self.positive_number.assign_add(positive_number)

        all_number = tf.size(y_true)
        all_number = tf.cast(all_number, tf.float32)
        self.all_number.assign_add(all_number)

        return self.result()

    def result(self):
        accuracy_rate = self.positive_number / self.all_number
        return accuracy_rate


class TrueConfidence(tf.keras.metrics.Metric):

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.positive_number = self.add_weight("positive_number", initializer="zeros")
        self.all_number = self.add_weight("all_number", initializer="zeros")

    def update_state(self, y_true, y_pred, *args, **kwargs):
        y_true = tf.where(y_true == -1, 0.0, y_true)
        y_pred = y_pred > self.threshold
        y_pred = tf.cast(y_pred, tf.float32)
        sum = y_pred + y_true
        is_equal = sum == 2
        is_equal = tf.cast(is_equal, tf.float32)
        positive_number = tf.reduce_sum(is_equal)
        self.positive_number.assign_add(positive_number)

        all_number = tf.reduce_sum(y_true)
        self.all_number.assign_add(all_number)

        return self.result()

    def result(self):
        accuracy_rate = self.positive_number / self.all_number
        return accuracy_rate


class FalseConfidence(tf.keras.metrics.Metric):

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.positive_number = self.add_weight("positive_number", initializer="zeros")
        self.all_number = self.add_weight("all_number", initializer="zeros")

    def update_state(self, y_true, y_pred, *args, **kwargs):
        y_true = tf.where(y_true == -1, 0.0, y_true)
        y_pred = y_pred > self.threshold
        y_pred = tf.cast(y_pred, tf.float32)
        sum = y_pred + y_true
        is_equal = sum == 0
        is_equal = tf.cast(is_equal, tf.float32)
        positive_number = tf.reduce_sum(is_equal)
        self.positive_number.assign_add(positive_number)

        all_number = tf.reduce_sum(1 - y_true)
        self.all_number.assign_add(all_number)

        return self.result()

    def result(self):
        accuracy_rate = self.positive_number / self.all_number
        return accuracy_rate


class Categorical(tf.keras.metrics.Metric):

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.positive_number = self.add_weight("positive_number", initializer="zeros")
        self.all_number = self.add_weight("all_number", initializer="zeros")

    def update_state(self, y_true, y_pred, *args, **kwargs):
        is_object = y_true[..., 0] != -1
        is_object = tf.cast(is_object, tf.float32)

        y_pred_argmax = tf.argmax(y_pred, -1)
        y_pred_argmax = tf.cast(y_pred_argmax, tf.float32)

        positive = y_pred_argmax == y_true[..., 0]
        positive = tf.cast(positive, tf.float32)

        positive = positive * is_object
        positive_number = tf.reduce_sum(positive)
        self.positive_number.assign_add(positive_number)

        all_number = tf.reduce_sum(is_object)
        self.all_number.assign_add(all_number)

        return self.result()

    def result(self):
        accuracy_rate = self.positive_number / self.all_number
        return accuracy_rate


class Precision(tf.keras.metrics.Metric):

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.positive_number = self.add_weight("positive_number", initializer="zeros")
        self.all_number = self.add_weight("all_number", initializer="zeros")

    def get_iou(self, y_true, y_pred):
        y_true_half = y_true[..., 2:4] / 2
        y_pred_half = y_pred[..., 2:4] / 2
        boxes1_pt1 = y_true[..., 0:2] - y_true_half
        boxes1_pt2 = y_true[..., 0:2] + y_true_half

        boxes2_pt1 = y_pred[..., 0:2] - y_pred_half
        boxes2_pt2 = y_pred[..., 0:2] + y_pred_half

        boxes1 = tf.concat((boxes1_pt1, boxes1_pt2), -1)
        boxes2 = tf.concat((boxes2_pt1, boxes2_pt2), -1)

        is_object = y_true[..., 2] > 0
        is_object = tf.cast(is_object, tf.float32)

        iou = c_functions.calc_iou(boxes1, boxes2)
        iou = iou * is_object

        positive = iou > self.threshold
        positive = tf.cast(positive, tf.float32)
        return positive

    def get_cate(self, y_true, y_pred):
        is_object = y_true[..., 0] != -1
        is_object = tf.cast(is_object, tf.float32)

        y_pred_argmax = tf.argmax(y_pred, -1)
        y_pred_argmax = tf.cast(y_pred_argmax, tf.float32)

        positive = y_pred_argmax == y_true[..., 0]
        positive = tf.cast(positive, tf.float32)

        positive = positive * is_object
        return positive

    def update_state(self, y_true, y_pred, *args, **kwargs):
        is_object = y_true[..., 4] == 1
        is_object = tf.cast(is_object, tf.float32)

        iou = self.get_iou(y_true[..., 0:4], y_pred[..., 0:4])
        cate = self.get_cate(y_true[..., 5:], y_pred[..., 5:])

        pred_positive = y_pred[..., 4] > self.threshold
        pred_positive = tf.cast(pred_positive, tf.float32)

        sum = iou + pred_positive + cate + is_object
        positive = sum == 4.0
        positive = tf.cast(positive, tf.float32)
        positive_number = tf.reduce_sum(positive)
        all_number = tf.reduce_sum(pred_positive)

        self.positive_number.assign_add(positive_number)
        self.all_number.assign_add(all_number)

        return self.result()

    def result(self):
        accuracy_rate = self.positive_number / self.all_number
        return accuracy_rate


class Recall(Precision):

    def __init__(self, threshold=0.3):
        super().__init__()
        self.threshold = threshold
        self.positive_number = self.add_weight("positive_number", initializer="zeros")
        self.all_number = self.add_weight("all_number", initializer="zeros")

    def update_state(self, y_true, y_pred, *args, **kwargs):
        is_object = y_true[..., 4] == 1
        is_object = tf.cast(is_object, tf.float32)

        iou = self.get_iou(y_true[..., 0:4], y_pred[..., 0:4])
        cate = self.get_cate(y_true[..., 5:], y_pred[..., 5:])

        pred_positive = y_pred[..., 4] > self.threshold
        pred_positive = tf.cast(pred_positive, tf.float32)

        sum = iou + pred_positive + cate + is_object
        positive = sum == 4.0
        positive = tf.cast(positive, tf.float32)
        positive_number = tf.reduce_sum(positive)

        all_number = tf.reduce_sum(is_object)

        self.positive_number.assign_add(positive_number)
        self.all_number.assign_add(all_number)

        return self.result()

    def result(self):
        accuracy_rate = self.positive_number / self.all_number
        return accuracy_rate


class MAP(tf.keras.metrics.Metric):

    def __init__(self, class_num, iou_threshold=0.5, score_threshold=0.5, nms_max_output_size=100,
                 nms_iou_threshold=0.5,
                 nms_score_shresold=0.8):
        super().__init__()
        self.class_num = class_num
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.nms_max_output_size = nms_max_output_size
        self.nms_iou_threshold = nms_iou_threshold
        self.nms_score_shresold = nms_score_shresold

        # self.true_positive_num = None
        # self.pred_positive_num = None
        # self.true_num = None

        self.true_positive_num = tf.Variable(tf.zeros((self.class_num,)))
        self.pred_positive_num = tf.Variable(tf.ones((self.class_num,)) * 1e-8)
        self.true_num = tf.Variable(tf.ones((self.class_num,)) * 1e-8)

        self.reset_variable()

    def reset_variable(self):

        self.true_positive_num.assign(tf.zeros((self.class_num,)))
        self.pred_positive_num.assign((tf.ones((self.class_num,)) * 1e-8))
        self.true_num.assign(tf.ones((self.class_num,)) * 1e-8)

    def update_map(self, y_true, y_pred):

        true_positive_num = tf.zeros((self.class_num,))
        pred_positive_num = tf.zeros((self.class_num,))
        true_num = tf.zeros((self.class_num,))

        for true_item, pred_item in zip(y_true, y_pred):
            indices = tf.image.non_max_suppression(pred_item[..., 0:4], pred_item[..., 4],
                                                   self.nms_max_output_size, self.nms_iou_threshold,
                                                   self.nms_score_shresold)
            pred_positive = tf.gather(pred_item, indices)

            pred_positive_cate = tf.argmax(pred_positive[..., 5:], -1)
            values, _, counts = tf.unique_with_counts(pred_positive_cate)
            indices = values[..., tf.newaxis]
            counts = tf.cast(counts, tf.float32)
            pred_positive_num = tf.tensor_scatter_nd_add(pred_positive_num, indices, counts)

            true_positive = tf.boolean_mask(true_item, true_item[..., 4] == 1)
            updates = [1.0]
            for item in true_positive:
                ious = c_functions.calc_iou(item[0:4], pred_positive[..., 0:4])
                ious = ious > self.iou_threshold
                cates = item[..., 5] == tf.cast(tf.argmax(pred_positive[..., 5:], -1), tf.float32)
                positive = ious & cates
                if tf.reduce_any(positive):
                    indices = tf.cast(item[5], tf.int32)
                    indices = indices[tf.newaxis, tf.newaxis, ...]
                    true_positive_num = tf.tensor_scatter_nd_add(true_positive_num, indices, updates)

        is_object = y_true[..., 0] != -1
        has_object = tf.boolean_mask(y_true, is_object)
        has_object_cate = has_object[..., 5]
        values, _, counts = tf.unique_with_counts(has_object_cate)
        indices = values[..., tf.newaxis]
        indices = tf.cast(indices, tf.int32)
        counts = tf.cast(counts, tf.float32)
        true_num = tf.tensor_scatter_nd_add(true_num, indices, counts)

        return true_positive_num, pred_positive_num, true_num

    def update_state(self, y_true, y_pred, *args, **kwargs):
        true_positive_num, pred_positive_num, true_num = tf.py_function(self.update_map,
                                                                        [y_true, y_pred],
                                                                        [tf.float32, tf.float32,
                                                                         tf.float32])
        self.true_positive_num.assign_add(true_positive_num)
        self.pred_positive_num.assign_add(pred_positive_num)
        self.true_num.assign_add(true_num)

    def reset_states(self):
        precision = self.true_positive_num / self.pred_positive_num
        recall = self.true_positive_num / self.true_num
        ap = precision * recall
        # map, ap, recall, precision = self.result(True)
        tf.print("\nap:\n", ap.numpy().tolist())
        tf.print("\nrecall:\n", recall.numpy().tolist())
        tf.print("\nprecision:\n", precision.numpy().tolist())
        self.reset_variable()

    def result(self):
        precision = self.true_positive_num / self.pred_positive_num
        recall = self.true_positive_num / self.true_num
        ap = precision * recall
        map = tf.reduce_mean(ap)
        return map


if __name__ == '__main__':
    import convert_data_2 as l_datasets

    gt_file = "/opt/datasets/wider_face/wider_face_split/wider_face_val_bbx_gt.txt"
    image_path = "/opt/datasets/wider_face/WIDER_val/images"
    train_ds = l_datasets.WiderFaceDataSet(gt_file, image_path, batch_size=5598, target_size=(224, 224))
    file_names = []
    locations = []
    for file_name, location in train_ds:
        file_names.append(file_name)
        locations.append(location)

    parse = l_datasets.Parse(target_size=(224, 224))

    train_ds = tf.data.Dataset.from_tensor_slices((file_names, locations))
