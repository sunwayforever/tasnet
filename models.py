#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2020-11-17 14:21
import tensorflow as tf
import numpy as np
from config import *

from tensorflow import keras

from tensorflow.keras import layers, losses, metrics, optimizers, models

if tf.__version__ >= "2.3.0":
    tf.config.run_functions_eagerly(True)

# count = 0
epislon = 1e-9


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
            conv_layers.append(layers.LayerNormalization())
            conv_layers.append(
                # [M,B,K]
                layers.SeparableConv1D(
                    filters=B,
                    kernel_size=P,
                    strides=1,
                    padding="same",
                    dilation_rate=2 ** i,
                )
            )
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
        # ========== encoder
        self.encoder = layers.Conv1D(
            filters=N, kernel_size=L, strides=L // 2, activation="relu", name="encoder"
        )

        # ========== separator
        self.separation = keras.Sequential()
        self.separation.add(layers.Conv1D(B, 1, 1))
        for _ in range(R):
            self.separation.add(TemporalBlock())
        self.separation.add(layers.Conv1D(C * N, 1, 1))
        self.separation.add(layers.PReLU(shared_axes=[1]))
        self.separation.add(layers.Reshape((K, C, N)))
        self.separation.add(layers.Permute((2, 1, 3), input_shape=(K, C, N)))
        self.separation.add(layers.Softmax(axis=1))

        # ========== decoder
        self.decoder = keras.Sequential()
        self.decoder.add(layers.Dense(L, use_bias=False))
        self.decoder.add(
            layers.Lambda(
                lambda signal: tf.signal.overlap_and_add(
                    tf.reshape(signal, (-1, C, K, L)),
                    frame_step=L // 2,
                ),
            )
        )

    def model(self):
        input = layers.Input(shape=(SAMPLE_FRAMES))

        output = tf.expand_dims(input, axis=-1)
        # [M,K,N]
        output = self.encoder(output)
        print("encoder output:", output.shape)
        encoded_input = output
        # [M,K,B]

        # [M,C,K,N]
        output = self.separation(output)
        print("separation output:", output.shape)

        encoded_input = tf.expand_dims(encoded_input, 1)
        output = output * encoded_input
        print("mask output:", output.shape)

        output = self.decoder(output)
        print("decoder output:", output.shape)
        return keras.Model(input, output)

    @staticmethod
    def _calc_sdr(s, s_hat):
        def norm(x):
            return tf.reduce_sum(x ** 2, axis=-1, keepdims=True)

        s_target = tf.reduce_sum(s_hat * s, axis=-1, keepdims=True) * s / norm(s)
        e_noise = s_hat - s_target

        # global count
        # count += 1
        # print(f"check: {count}")
        # print("s")
        # print(s)
        # print("s_hat")
        # print(s_hat)
        # print("target")
        # print(s_target)
        # np.save(f"/tmp/s_{count}.npy", s)
        # np.save(f"/tmp/s_hat_{count}.npy", s_hat)
        # np.save(f"/tmp/target_{count}.npy", s_target)

        return (
            10
            * tf.math.log(norm(s_target) / (norm(e_noise) + epislon))
            / tf.math.log(10.0)
        )

    @staticmethod
    def loss(y, y_hat):
        sdr1 = TasNet._calc_sdr(y[:, 0], y_hat[:, 0]) + TasNet._calc_sdr(
            y[:, 1], y_hat[:, 1]
        )
        # sdr2 = TasNet._calc_sdr(y_hat[:, 1], y[:, 0]) + TasNet._calc_sdr(
        #     y_hat[:, 0], y[:, 1]
        # )
        # sdr = tf.maximum(sdr1, sdr2)
        return tf.reduce_mean(-sdr1) / C

    @staticmethod
    def mse_loss(y, y_hat):
        # return losses.MSE(y[:, 0], y_hat[:, 0]) + losses.MSE(y[:, 1], y_hat[:, 1])
        return losses.MSE(y, y_hat)
