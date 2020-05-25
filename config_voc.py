
board_log_dir = "board_log"
save_weight_file = "weights"


anchor_sizes = [[[14.000000000000002, 14.000000000000002], [22.000000000000007, 22.0], [28.0, 28.0]],
                [[42.0, 42.0], [56.00000000000001, 56.00000000000001], [84.00000000000001, 84.00000000000001]],
                [[111.99999999999997, 111.99999999999997], [200, 200], [250, 250]]]


image_target_size = (416, 416)


# filters基数 设为16可使所有层filters减少一半
filters = 32
grid_sizes = [52, 26, 13]
layer_num = len(grid_sizes)
train_label_file = "/opt/datasets/yymnist_yolo/train_labels.txt"
train_image_dir = "/opt/datasets/yymnist_yolo/train"
val_label_file = "/opt/datasets/yymnist_yolo/val_labels.txt"
val_image_dir = "/opt/datasets/yymnist_yolo/test"
class_num = 10
batch_size = 4

metric_iou_threshold = 0.5
metric_score_threshold = 0.8

nms_iou_threshold = 0.1
nms_score_threshold = metric_score_threshold
nms_max_output = 1000

cate_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog',
              'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

