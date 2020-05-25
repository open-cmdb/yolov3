
## 前言
一直都对目标检测很感兴趣 所以就自己从头开始实现了一个yolov3  路途还是有点艰辛😀😄 受环境等各方面因素目前仅在一些简单的数据上跑出了结果

## 特性
* 项目完整 数据预处理处理、anchor生成、预览数据、预览训练结果
* 使用最新版tensorflow2.2.0 模型部分纯tensorflow实现 应该是全程享受加速
* 模块化实现 结构简洁 可阅读性好
* 实时计算mAP
​
## 其它
* 解码用原论文公式解码长宽很容易出现NaN 所以我将其改为了2 * sigmoid(p)
* 动态计算置信度正负样本权重

## 训练yymnist
准备数据
```shell script
git clone https://github.com/YunYang1994/yymnist.git
mkdir -p /tmp/yymnist/train
mkdir -p /tmp/yymnist/val
python yymnist/make_data.py --images_num 1000 --images_path ./data/dataset/train --labels_txt ./data/dataset/yymnist_train.txt
python yymnist/make_data.py --images_num 200  --images_path ./data/dataset/test  --labels_txt ./data/dataset/yymnist_test.txt
```

## 转换label
tools -> convert_label_to_0_1.py

### 修改
```python
LABEL_FILE = "/opt/datasets/yymnist_yolo/yymnist_train.txt"
OUTPUT_FILE = "/opt/datasets/yymnist_yolo/train_labels.txt"
```

## 修改config中数据配置
```python
train_label_file = "/opt/datasets/yymnist_yolo/train_labels.txt"
train_image_dir = "/opt/datasets/yymnist_yolo/train"
val_label_file = "/opt/datasets/yymnist_yolo/val_labels.txt"
val_image_dir = "/opt/datasets/yymnist_yolo/test"
```

## 生成anchors(聚类可能会出现某个类为空的情况 结果就会为空 自己手动将其改为合适的值就好)
tools -> get_anchors.py

## 将生成的anchors配置的配置文件
```python
anchor_sizes = [[[14.000000000000002, 14.000000000000002], [22.000000000000007, 22.0], [28.0, 28.0]],
                [[42.0, 42.0], [56.00000000000001, 56.00000000000001], [84.00000000000001, 84.00000000000001]],
                [[111.99999999999997, 111.99999999999997], [200, 200], [250, 250]]]
```
## 预览训练数据(绿色表示真实框 红色表示对应的anchor)
tools -> dataset_previews
<p align="center">
    <img src="https://github.com/open-cmdb/yolov3/blob/master/images/yolov3-datasets-preview.png">
</p>

## 训练
train.py

```text
Epoch 00051: val_output_4_map did not improve from 0.79128
250/250 [==============================] - 138s 551ms/step - loss: 3.1002 - output_1_loss: 0.0591 - output_2_loss: 0.0308 - output_3_loss: 0.0059 - output_4_loss: 0.0000e+00 - output_1_location_1: 1.0000 - output_2_confidence_1: 0.9970 - output_2_true_confidence_1: 1.0000 - output_2_false_confidence_1: 0.9970 - output_3_categorical_1: 1.0000 - output_4_precision_1: 0.1617 - output_4_recall_1: 1.0000 - output_4_map: 0.9049 - val_loss: 3.4197 - val_output_1_loss: 0.0861 - val_output_2_loss: 0.0559 - val_output_3_loss: 0.2740 - val_output_4_loss: 0.0000e+00 - val_output_1_location_1: 1.0000 - val_output_2_confidence_1: 0.9972 - val_output_2_true_confidence_1: 0.9941 - val_output_2_false_confidence_1: 0.9972 - val_output_3_categorical_1: 0.9135 - val_output_4_precision_1: 0.1514 - val_output_4_recall_1: 0.9110 - val_output_4_map: 0.7884

ap:
 [0.7132132053375244,
 0.9144498109817505,
 0.8080149292945862,
 0.7785018086433411,
 0.8017507195472717,
 0.8155587315559387,
 0.8004550933837891,
 0.7287811040878296,
 0.7686083912849426,
 0.7549148201942444]

recall:
 [0.8333333134651184,
 0.9622641801834106,
 0.9189189076423645,
 0.8549618124961853,
 0.8991596698760986,
 0.9207921028137207,
 0.9351851940155029,
 0.8606557250022888,
 0.8333333134651184,
 0.8571428656578064]

precision:
 [0.8558558821678162,
 0.9503105878829956,
 0.8793103694915771,
 0.9105691313743591,
 0.8916666507720947,
 0.8857142925262451,
 0.8559321761131287,
 0.8467742204666138,
 0.9223300814628601,
 0.8807339668273926]
```

<p align="center">
    <img src="https://github.com/open-cmdb/yolov3/blob/master/images/yolov3-tensorboard.png">
</p>



## 指标说明
* output_1_loss: 坐标损失(GIOU)
* output_2_loss: 置信度损失(Binary cross entropy)
* output_3_loss: 分类损失(categorical cross entropy)
* output_4_loss: 为了和数据输出保持一致 没有任何意义

* output_1_location_1:        坐标准确率
* output_2_confidence_1:      总的置信度准确率
* output_2_true_confidence_1: 正样本置信度准确率
* output_2_false_confidence_1:负样本置信度准确率
* output_3_categorical_1:     分类准确率
* output_4_precision_1:       精准率（不考虑类别）
* output_4_recall_1:          召回率（不考虑类别）
* output_4_map:               mAP


## 预测
tools -> predict_test_images.py