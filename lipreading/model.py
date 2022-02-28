import os
import sys
from tcn import TCN, tcn_full_summary
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, metrics, regularizers
import matplotlib.pyplot as plt

# Adding the parent directory to the sys.path
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)
import config


def threeD_to_2D_tensor(x):
    n_batch, s_time, sx, sy, n_channels = x.shape
    if n_batch == None:
        n_batch = config.BATCH_SIZE
    return tf.reshape(x, (n_batch*s_time, sx, sy, n_channels))


class ResBlock(layers.Layer):
    def __init__(self, filter_nums, strides=1, residual_path=False):
        super(ResBlock, self).__init__()

        self.conv_1 = layers.Conv2D(
            filter_nums, (3, 3), strides=strides, padding='same')
        self.bn_1 = layers.BatchNormalization()
        self.act_relu = layers.Activation('relu')

        self.conv_2 = layers.Conv2D(
            filter_nums, (3, 3), strides=1, padding='same')
        self.bn_2 = layers.BatchNormalization()

        if strides != 1:
            self.block = Sequential()
            self.block.add(layers.Conv2D(filter_nums, (1, 1), strides=strides))
        else:
            self.block = lambda x: x

    def call(self, inputs, training=None):

        x = self.conv_1(inputs)
        x = self.bn_1(x, training=training)
        x = self.act_relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x, training=training)

        identity = self.block(inputs)
        outputs = layers.add([x, identity])
        outputs = tf.nn.relu(outputs)

        return outputs


class ResNet(keras.Model):
    def __init__(self, layers_dims, nums_class=10):
        super(ResNet, self).__init__()

        self.model = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1)),
                                 layers.BatchNormalization(),
                                 layers.Activation('relu'),
                                 layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')])

        self.layer_1 = self.ResNet_build(64, layers_dims[0])
        self.layer_2 = self.ResNet_build(128, layers_dims[1], strides=2)
        self.layer_3 = self.ResNet_build(256, layers_dims[2], strides=2)
        self.layer_4 = self.ResNet_build(512, layers_dims[3], strides=2)
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.fc_model = layers.Dense(nums_class)

    def call(self, inputs, training=None):
        x = self.model(inputs)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.avg_pool(x)
        x = self.fc_model(x)
        return x

    def ResNet_build(self, filter_nums, block_nums, strides=1):
        build_model = Sequential()
        build_model.add(ResBlock(filter_nums, strides))
        for _ in range(1, block_nums):
            build_model.add(ResBlock(filter_nums, strides=1))
        return build_model


def ResNet18():
    return ResNet([2, 2, 2, 2])


def ResNet34():
    return ResNet([3, 4, 6, 3])


class Lipreading(keras.Model):
    def __init__(self, input_shape=(120, 96, 96, 1)):
        super(Lipreading, self).__init__()
        self.input_spec = tf.keras.layers.InputSpec(
            shape=(None, 120, 96, 96, 1))
        self.frontend = Sequential([
            layers.Conv3D(1, (1, 1, 1), strides=(
                1, 1, 1), activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
            layers.BatchNormalization(),
            layers.MaxPool3D(),
            # layers.Dropout(0.5),
            layers.Conv3D(1, (1, 1, 1), strides=(
                1, 1, 1), activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
            layers.BatchNormalization(),
            layers.MaxPool3D(),
            # layers.Dropout(0.5),
            layers.Conv3D(1, (1, 1, 1), strides=(
                1, 1, 1), activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
            layers.BatchNormalization(),
            layers.MaxPool3D(),
            # layers.Dropout(0.5)
        ])

        self.trunk = ResNet18()
        self.trunk.build(input_shape=(75, 12, 12, 1))
        self.bn_01 = layers.BatchNormalization()
        self.backend_out = 10
        self.tcn = TCN(input_shape=(30, self.backend_out),
                       # self.tcn = TCN(input_shape=(15, self.backend_out),
                       nb_filters=20,
                       dropout_rate=0.2
                       )

    def call(self, x):
        B, T, H, W, C = x.shape
        if B == None:
            B = config.BATCH_SIZE
        # print(x.shape)
        x = self.frontend(x)
        # print(x.shape)
        # outpu should be B x Tnew x H x W x C2
        Tnew = x.shape[1]
        x = threeD_to_2D_tensor(x)
        # print(x.shape)
        x = self.trunk(x)
        x = self.bn_01(x)
        # print(x.shape)
        x = tf.reshape(x, (B, Tnew, x.shape[1]))
        # print(x.shape)
        x = self.tcn(x)
        # print(x.shape)
        return x
