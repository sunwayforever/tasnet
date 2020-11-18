#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2020-11-17 14:21
import tensorflow as tf
import numpy as np
from config import *

from tensorflow import keras

from tensorflow.keras import layers, losses, metrics, optimizers, models

# tf.config.run_functions_eagerly(True)


class TemporalBlock(layers.Layer):
    def __init__(self):
        super(TemporalBlock, self).__init__()
        self.layers = []

    def build(self, input_shape):
        # in_channels: B
        # out_channels: H
        # kernel: P
        for i in range(X):
            conv_layers = []
            # [M,H,K]
            conv_layers.append(layers.Conv1D(filters=H, kernel_size=1))
            conv_layers.append(layers.PReLU(shared_axes=[1]))
            conv_layers.append(
                # [M,B,K]
                layers.Conv1D(filters=B, kernel_size=1, strides=1, padding="same")
            )
            conv_layers.append(layers.PReLU(shared_axes=[1]))
            self.layers.append(conv_layers)

    def call(self, input):
        output = input
        for conv_layers in self.layers:
            input = output
            for layer in conv_layers:
                output = layer(output)
            output += input
        return output


class TasNet:
    def __init__(self):
        self.encoder = layers.Conv1D(
            filters=N, kernel_size=L, strides=L // 2, activation="relu", name="encoder"
        )
        self.decoder = keras.Sequential()
        self.decoder.add(layers.Dense(L, use_bias=False))
        self.decoder.add(
            layers.Lambda(
                lambda signal: tf.signal.overlap_and_add(
                    tf.reshape(signal, (BATCH_SIZE, K, L)),
                    frame_step=L // 2,
                ),
            )
        )

        self.bottleneck = layers.Conv1D(B, 1, 1)
        self.separation_blocks = [TemporalBlock() for _ in range(R)]
        self.separation_conv = [layers.Conv1D(N, 1, 1) for _ in range(C)]

    def model(self):
        input = layers.Input(shape=(SAMPLE_FRAMES))

        output = tf.expand_dims(input, axis=-1)
        # [M,N,K]
        output = self.encoder(output)
        print("encoder output:", output.shape)

        encoded_input = output
        # [M,B,K]
        output = self.bottleneck(output)

        for conv_block in self.separation_blocks:
            output = conv_block(output)
        print("tempora conv output:", output.shape)

        outputs = [block(output) for block in self.separation_conv]

        # [M,2N,K]
        print("separation output:", outputs[0].shape)

        probs = tf.nn.softmax(tf.stack(outputs, axis=-1))
        probs = tf.unstack(probs, axis=-1)
        outputs = [mask * encoded_input for mask in probs]

        outputs = tf.stack([self.decoder(output) for output in outputs])
        return keras.Model(input, outputs)

    @staticmethod
    def _calc_sdr(y, y_hat):
        def norm(x):
            return tf.reduce_sum(x ** 2, axis=-1, keepdims=True)

        y_target = tf.reduce_sum(y_hat * y, axis=-1, keepdims=True) * y / norm(y)
        up = norm(y_target)
        low = norm(y_hat - y_target) + 1e-10
        return 10 * tf.math.log((up / low) + 1e-10) / tf.math.log(10.0)

    @staticmethod
    def loss(y, y_hat):
        sdr1 = TasNet._calc_sdr(y_hat[0], y[0]) + TasNet._calc_sdr(y_hat[1], y[1])
        sdr2 = TasNet._calc_sdr(y_hat[1], y[0]) + TasNet._calc_sdr(y_hat[0], y[1])
        sdr = tf.maximum(sdr1, sdr2)
        return tf.reduce_mean(-sdr) / C
