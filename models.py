import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


class Convolutional(tf.keras.Model):

    def __init__(self, filters, kernel_size, downsample=False, activate=True, bn=True):
        super().__init__()
        self.downsample = downsample
        self.activate = activate
        self.bn = bn
        self.m_layers = []
        padding = "valid" if downsample else "same"
        strides = 2 if downsample else 1
        self.m_layers = []
        if self.downsample:
            input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))
            self.m_layers.append(input_layer)
        conv_layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                            padding=padding,
                                            use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                            bias_initializer=tf.constant_initializer(0.))
        self.m_layers.append(conv_layer)
        if self.bn:
            bn_layer = tf.keras.layers.BatchNormalization()
            self.m_layers.append(bn_layer)

    def call(self, inputs, training=None, mask=None, **kwargs):
        x = inputs
        for layer in self.m_layers:
            x = layer(x, training=training)
        if self.activate:
            x = tf.nn.leaky_relu(x, alpha=0.1)
        return x


class ResBlock(tf.keras.Model):

    def __init__(self, input_shape):
        super().__init__()
        if input_shape[-1] % 2 != 0:
            raise Exception("input_shape[-1] % 2 != 0")
        self.m_conv_1 = Convolutional(int(input_shape[-1] / 2), 1)
        self.m_conv_2 = Convolutional(input_shape[-1], 3)

    def call(self, inputs, training=None, mask=None, **kwargs):
        x = inputs
        x = self.m_conv_1(x, training=training)
        x = self.m_conv_2(x, training=training)
        return inputs + x


class ResOperator(tf.keras.Model):

    def __init__(self, input_shape, res_block_num):
        super().__init__()
        self.m_input_layer = Convolutional(input_shape[-1] * 2, 3, downsample=True)
        res_block_input_shape = [int(input_shape[0] / 2), int(input_shape[1] / 2), input_shape[2] * 2]
        self.res_block_layers = [ResBlock(res_block_input_shape) for _ in range(res_block_num)]

    def call(self, inputs, training=None, mask=None, **kwargs):
        x = self.m_input_layer(inputs, training=training)
        for layer in self.res_block_layers:
            x = layer(x, training=training)
        return x


class ConvolutionalSet(tf.keras.Model):

    def __init__(self, filters):
        super().__init__()
        self.m_layers = [
            Convolutional(filters, 1),
            Convolutional(filters * 2, 3),

            Convolutional(filters, 1),
            Convolutional(filters * 2, 3),

            Convolutional(filters, 1)
        ]

    def call(self, inputs, training=None, mask=None, **kwargs):
        x = inputs
        for layer in self.m_layers:
            x = layer(x, training=training)
        return x


class Darknet53(tf.keras.Model):

    def __init__(self, filters):
        super().__init__()
        self.filters = filters
        self.branch_1_layers = [
            Convolutional(filters * 1, 3),
            ResOperator(input_shape=(416, 416, filters * 1), res_block_num=1),
            ResOperator(input_shape=(208, 208, filters * 2), res_block_num=2),
            ResOperator(input_shape=(104, 104, filters * 4), res_block_num=8),
            ResOperator(input_shape=(52, 52, filters * 8), res_block_num=8),
            ResOperator(input_shape=(26, 26, filters * 16), res_block_num=4)
        ]

    def call(self, inputs, training=True):
        x = inputs
        branch_1 = None
        branch_2 = None
        branch_3 = None
        for index, layer in enumerate(self.branch_1_layers):
            x = layer(x, training=training)
            if index == 3:
                branch_3 = x
            elif index == 4:
                branch_2 = x
            elif index == 5:
                branch_1 = x
        return branch_3, branch_2, branch_1


class YoloV3(tf.keras.Model):
    grad_sizes = tf.constant([32, 16, 8])
    size = tf.constant([256, 256])

    def __init__(self, filters, anchors, grid_sizes, class_num):
        super().__init__()
        self.filters = filters
        self.anchors = anchors
        self.grid_sizes = grid_sizes
        self.class_num = class_num
        self.output_filters = (class_num + 5) * 3
        output_nums = tf.pow(grid_sizes, 2).numpy() * 3

        self.o_shape1 = [-1] + anchors[0].shape[:-1].as_list() + [5+self.class_num]
        self.o_shape2 = [-1] + anchors[1].shape[:-1].as_list() + [5+self.class_num]
        self.o_shape3 = [-1] + anchors[2].shape[:-1].as_list() + [5+self.class_num]

        self.output_shape1 = [-1, output_nums[0], 5+self.class_num]
        self.output_shape2 = [-1, output_nums[1], 5+self.class_num]
        self.output_shape3 = [-1, output_nums[2], 5+self.class_num]
        # self.output_shape = tf.TensorShape(output_shape)

        self.backbone = Darknet53(filters * 1)
        self.branch_1_layers = [
            ConvolutionalSet(filters=filters * 16),
            Convolutional(filters * 32, 3),
            Convolutional(self.output_filters, 1, activate=False, bn=False),
        ]

        self.branch_2_layers = [
            ConvolutionalSet(filters=filters * 8),
            Convolutional(filters * 16, 3),
            Convolutional(self.output_filters, 1, activate=False, bn=False),
        ]

        self.branch_3_layers = [
            ConvolutionalSet(filters=filters * 4),
            Convolutional(filters * 8, 3),
            Convolutional(self.output_filters, 1, activate=False, bn=False),
        ]

        self.branch_1_to_2_layers = [
            Convolutional(filters * 8, 1),
            tf.keras.layers.UpSampling2D(2)
        ]

        self.branch_2_to_3_layers = [
            Convolutional(filters * 4, 1),
            tf.keras.layers.UpSampling2D(2)
        ]

    def decode(self, layer_inputs, layer_index):
        boxes_xy = layer_inputs[..., 0:2]
        boxes_wh = layer_inputs[..., 2:4]
        conf = layer_inputs[..., 4:5]
        cate = layer_inputs[..., 5:]

        grid_size = self.grid_sizes[layer_index]
        boxes_xy = tf.sigmoid(boxes_xy)
        boxes_xy = boxes_xy * 1.0 / grid_size
        anchors_wh = self.anchors[layer_index][..., 2:4] - self.anchors[layer_index][..., 0:2]
        boxes_wh = anchors_wh * tf.sigmoid(boxes_wh) * 2

        indices = tf.linspace(0.0, 1.0 - 1.0 / grid_size, grid_size)
        xx, yy = tf.meshgrid(indices, indices)
        grid = tf.stack((yy, xx), -1)
        original_shape = grid.shape
        grid = tf.tile(grid, [1, 1, 3])
        new_shape = original_shape[:-1] + [3] + original_shape[-1:]
        grid = tf.reshape(grid, new_shape)
        boxes_xy = boxes_xy + grid

        boxes_wh_half = boxes_wh / 2
        pt1 = boxes_xy - boxes_wh_half
        pt2 = boxes_xy + boxes_wh_half

        # pt1 = pt1 / pt1
        # pt2 = pt2 / pt2
        #
        # pt1 = pt1 * self.anchors[layer_index][..., 1:2]
        # pt2 = pt2 * self.anchors[layer_index][..., 2:4]

        conf = tf.sigmoid(conf)
        cate = tf.nn.softmax(cate)

        decoded = tf.concat((pt1, pt2, conf, cate), -1)

        return decoded

    # def _get_predict_with_nms_single(self, loca, conf, cate):
    #     indices = tf.image.non_max_suppression(loca, conf[..., 0], self.nms_max_output, self.nms_iou_threshold,
    #                                            self.nms_score_threshold)
    #     nms_loca = tf.gather(loca, indices)
    #     nms_conf = tf.gather(conf, indices)
    #     nms_cate = tf.gather(cate, indices)
    #
    #     nms_cate = tf.argmax(nms_cate, -1)
    #     result = tf.concat((nms_loca, nms_conf, nms_cate), -1)
    #     return result

    # def predict_with_nms(self, images):
    #     y_pred = self.predict(images)
    #     pred = []
    #     for image, loca, conf, cate in zip(images, y_pred[0], y_pred[1], y_pred[2]):
    #         result = self._get_predict_with_nms_single(loca, conf, cate)
    #         pred.append(result)
    #     return pred

    def call(self, inputs, training=True):
        x = inputs
        branch_3_input_1, branch_2_input_1, branch_1_input_1 = self.backbone(x, training=training)

        branch_1_input = branch_1_input_1
        branch_1_output = branch_1_input
        for index, layer in enumerate(self.branch_1_layers):
            branch_1_output = layer(branch_1_output, training=training)
            if index == 0:
                branch_2_input_2 = branch_1_output

        for layer in self.branch_1_to_2_layers:
            branch_2_input_2 = layer(branch_2_input_2, training=training)

        branch_2_input = tf.concat((branch_2_input_1, branch_2_input_2), -1)
        branch_2_output = branch_2_input
        for index, layer in enumerate(self.branch_2_layers):
            branch_2_output = layer(branch_2_output, training=training)
            if index == 0:
                branch_3_input_2 = branch_2_output

        for layer in self.branch_2_to_3_layers:
            branch_3_input_2 = layer(branch_3_input_2, training=training)

        branch_3_input = tf.concat((branch_3_input_1, branch_3_input_2), -1)
        branch_3_output = branch_3_input
        for index, layer in enumerate(self.branch_3_layers):
            branch_3_output = layer(branch_3_output, training=training)

        branch_3_output = tf.reshape(branch_3_output, self.o_shape1)
        branch_2_output = tf.reshape(branch_2_output, self.o_shape2)
        branch_1_output = tf.reshape(branch_1_output, self.o_shape3)

        branch_3_decoded = self.decode(branch_3_output, 0)
        branch_2_decoded = self.decode(branch_2_output, 1)
        branch_1_decoded = self.decode(branch_1_output, 2)

        # shape = [-1] , , self.class_num + 5]

        branch_3_decoded = tf.reshape(branch_3_decoded, self.output_shape1)
        branch_2_decoded = tf.reshape(branch_2_decoded, self.output_shape2)
        branch_1_decoded = tf.reshape(branch_1_decoded, self.output_shape3)

        decoded = tf.concat((branch_3_decoded, branch_2_decoded, branch_1_decoded), 1)

        return decoded[..., 0:4], decoded[..., 4:5], decoded[..., 5:], decoded


if __name__ == '__main__':
    writer = tf.summary.create_file_writer("/tmp/yolo")
    writer.as_default()
    tf.summary.trace_on()

    model = YoloV3(None, 20)
    input = tf.ones((1, 416, 416, 3))
    r = model(input)
    print("")
    tf.summary.trace_export(
        name="my_func_trace",
        step=0,
        profiler_outdir="/tmp/yolo")

    # l1 = ConvolutionalSet(3)
    # r = l1(input, training=True)
    # pass
