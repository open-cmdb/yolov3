import tensorflow as tf


def calc_iou(boxes1, boxes2, ret_area=False):
    top = tf.maximum(boxes1[..., 0], boxes2[..., 0])
    left = tf.maximum(boxes1[..., 1], boxes2[..., 1])
    bottom = tf.minimum(boxes1[..., 2], boxes2[..., 2])
    right = tf.minimum(boxes1[..., 3], boxes2[..., 3])

    height = bottom - top
    width = right - left

    height = tf.where(height > 0, height, 0)
    width = tf.where(width > 0, width, 0)

    intersection_area = height * width
    area_1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area_2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union_area = area_1 + area_2 - intersection_area

    iou = intersection_area / union_area
    if ret_area:
        return iou, intersection_area, union_area
    return iou


def calc_giou(boxes1, boxes2):
    iou, intersection_area, union_area = calc_iou(boxes1, boxes2, True)

    left = tf.minimum(boxes1[..., 1], boxes2[..., 1])
    top = tf.minimum(boxes1[..., 0], boxes2[..., 0])
    right = tf.maximum(boxes1[..., 3], boxes2[..., 3])
    bottom = tf.maximum(boxes1[..., 2], boxes2[..., 2])

    width = right - left
    height = bottom - top

    area = width * height
    giou = iou - (area - union_area) / area

    return giou


if __name__ == '__main__':
    a = tf.constant([[20, 20, 10, 10]], tf.float32)
    b = tf.constant([[0, 0, 0, 0]], tf.float32)
    giou = calc_giou(a, b)
    print(giou.numpy())
