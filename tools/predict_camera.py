import os
import sys
import datetime

import cv2
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

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

anchor_sizes = tf.convert_to_tensor(l_config.anchor_sizes, tf.float32)
anchor_sizes = anchor_sizes / l_config.image_target_size[0]
anchor_instance = l_anchors.Anchor(anchor_sizes, l_config.grid_sizes, l_config.image_target_size)
anchors = anchor_instance.get_anchors()

train_parse = l_datasets.Parse(l_config.train_image_dir, anchors, l_config.grid_sizes, l_config.image_target_size)
val_parse = l_datasets.Parse(l_config.val_image_dir, anchors, l_config.grid_sizes, l_config.image_target_size)

train_ds = tf.data.TextLineDataset(l_config.train_label_file)
train_ds = train_ds.map(train_parse)
train_ds = train_ds.shuffle(128)
train_ds = train_ds.batch(l_config.batch_size)
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

val_ds = tf.data.TextLineDataset(l_config.val_label_file)
val_ds = val_ds.map(val_parse)
val_ds = val_ds.batch(l_config.batch_size)
val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

model = l_models.YoloV3(l_config.filters, anchors, l_config.grid_sizes, l_config.class_num)

loca_loss = l_losses.LocationLoss(anchors)
conf_loss = l_losses.ConfidenceLoss()
cate_loss = l_losses.CategoricalLoss()
all_loss = l_losses.AllLoss()

loca_metric = l_metrics.Location()
conf_metric = l_metrics.Confidence()
true_conf_metric = l_metrics.TrueConfidence()
false_conf_metric = l_metrics.FalseConfidence()
cate_metric = l_metrics.Categorical()

precision = l_metrics.Precision()
recall = l_metrics.Recall()

optimizer = tf.keras.optimizers.SGD(1e-3, momentum=0.9)

item = next(iter(train_ds))
pred = model(item[0])

# checkpoint = tf.keras.callbacks.ModelCheckpoint(l_config.SAVE_WEIGHT_FILE)
# tensor_board = tf.keras.callbacks.TensorBoard(l_config.BOARD_LOG_DIR, update_freq=10)


model.load_weights(l_config.save_weight_file)


def draw(image, loca, conf, cate, size):
    images = image[tf.newaxis, ...]

    indices = tf.image.non_max_suppression(loca, conf[..., 0], l_config.nms_max_output, l_config.nms_iou_threshold,
                                           l_config.nms_score_threshold)
    nms_loca = tf.gather(loca, indices)
    nms_conf = tf.gather(conf, indices)
    nms_cate = tf.gather(cate, indices)

    nms_cate = tf.argmax(nms_cate, -1)

    images = tf.image.draw_bounding_boxes(images, [nms_loca], [[255, 0, 0]])
    image = images[0]
    image = tf.cast(image, tf.int32)
    image = image.numpy()

    cv2_loca = nms_loca.numpy()[..., :2] * np.array(size)
    cv2_loca = cv2_loca.astype(np.int32)
    cv2_loca[..., 0] += 10
    cv2_cate = nms_cate.numpy().astype(np.int32)
    for index, (loca, cate) in enumerate(zip(cv2_loca, cv2_cate)):
        loca = tuple(reversed(loca))
        cate_name = l_config.cate_names[cate]
        image = cv2.putText(image, cate_name, loca, cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))
    return image


@tf.function
def predict(images):
    y_pred = model(images, training=False)
    return y_pred


if __name__ == '__main__':
    # cap = cv2.VideoCapture("/opt/videos/one.mov")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("video error")
        exit(1)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    video_writer = cv2.VideoWriter("/tmp/test24.mp4", fourcc, fps, (width, height))
    while True:
        success, original_image = cap.read()
        if success is False:
            break
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        image = tf.image.resize(original_image, l_config.image_target_size)
        image = tf.cast(image, tf.float32)
        images = image[tf.newaxis, ...]
        images = images / 127.5 - 1
        y_pred = predict(images)
        image = draw(original_image, y_pred[0][0], y_pred[1][0], y_pred[2][0], (height, width))
        image = image.astype(np.uint8)
        image = image[..., [2, 1, 0]]
        cv2.imshow("results", image)
        video_writer.write(image)
        print("a")
    cap.release()
    video_writer.release()





