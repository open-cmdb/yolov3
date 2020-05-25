import os
import sys
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

train_ds = tf.data.TextLineDataset(l_config.train_label_file)
train_ds = train_ds.map(train_parse)
train_ds = train_ds.shuffle(128)
train_ds = train_ds.batch(l_config.batch_size)
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

val_ds = tf.data.TextLineDataset(l_config.val_label_file)
val_ds = val_ds.map(val_parse)
val_ds = val_ds.batch(l_config.batch_size)
val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

model = l_models.YoloV3(anchors, l_config.grid_sizes, l_config.class_num, is_decode=True)

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

item = next(iter(train_ds))
pred = model(item[0])


optimizer = tf.keras.optimizers.SGD(1e-4, momentum=0.9)

checkpoint = tf.keras.callbacks.ModelCheckpoint(l_config.save_weight_file, save_best_only=True)
tensor_board = tf.keras.callbacks.TensorBoard(l_config.board_log_dir, update_freq=10)

# tf.config.experimental_run_functions_eagerly(True)
# model.evaluate(val_ds)

model.load_weights(l_config.save_weight_file)


for nms_iou_threshold in range(1, 9):
    nms_iou_threshold /= 10

    for nms_score_threshold in range(1, 9):
        nms_score_threshold /= 10

        map = l_metrics.MAP(l_config.class_num,
                            l_config.metric_iou_threshold,
                            l_config.metric_score_threshold,
                            l_config.nms_max_output,
                            nms_iou_threshold,
                            nms_score_threshold)

        model.compile(optimizer=optimizer,
                      loss=[loca_loss, conf_loss, cate_loss, all_loss],
                      metrics=[[], [], [],
                               [map]])
        print(f"nms_iou_threshold: {nms_iou_threshold} nms_score_threshold: {nms_score_threshold}")
        model.evaluate(val_ds)
        # model.fit(train_ds, epochs=100, initial_epoch=initial_epoch,
        #           validation_data=val_ds, callbacks=[tensor_board, checkpoint])
