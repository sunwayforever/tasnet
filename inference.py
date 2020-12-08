#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2020-10-29 08:33
import tensorflow as tf
import numpy as np
import os
import argparse
import glob
import re

import soundfile as sf
from tensorflow import keras
from tensorflow.keras import layers, losses, metrics, optimizers, models

from config import *
from models import *

FLAGS = None


def norm(x):
    return (x ** 2).sum()


def inference(model):
    clean, _ = sf.read(FLAGS.clean, dtype="float32")
    noise, _ = sf.read(FLAGS.noise, dtype="float32")

    rem = noise.shape[0] % SAMPLE_FRAMES

    if rem != 0:
        noise = noise[:-rem]

    clean = clean[: noise.size]

    noisy = clean + np.sqrt(norm(clean) / norm(noise)) * noise
    x = model.predict(np.reshape(noisy, (-1, SAMPLE_FRAMES)))

    clean_hat = np.reshape(x[:, 0, :], (-1))
    noise_hat = np.reshape(x[:, 1, :], (-1))

    print("sdr of clean:", TasNet._calc_sdr(clean, clean_hat))
    print("sdr of noise:", TasNet._calc_sdr(noise, noise_hat))

    sf.write("/tmp/noisy.wav", noisy, 16000)
    sf.write("/tmp/clean.wav", clean_hat, 16000)
    sf.write("/tmp/noise.wav", noise_hat, 16000)


if __name__ == "__main__":
    model = keras.models.load_model(
        SAVED_MODEL_PATH, custom_objects={"loss": TasNet.loss}
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("clean")
    parser.add_argument("noise")
    FLAGS = parser.parse_args()
    inference(model)
