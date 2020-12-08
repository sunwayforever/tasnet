#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2020-10-28 13:35
import io
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
from config import *
import pickle
import time

with open(f"{DATA_DIR}/data.pkl", "rb") as f:
    data = pickle.load(f)

MIN_ENERGY = 1e-3


def get_dataset(mode):
    return (
        tf.data.Dataset.from_generator(
            lambda: data_generator(mode),
            output_types=(tf.string, tf.string),
        )
        .map(decode_op, num_parallel_calls=40)
        .filter(filter_op)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )


def norm(x):
    return (x ** 2).sum()


def filter_op(a, b):
    return tf.numpy_function(filter_audio, [a, b], tf.bool)


def filter_audio(a, b):
    return norm(a) > MIN_ENERGY


def decode_op(a, b):
    return tf.numpy_function(decode_audio, [a, b], (tf.float32, tf.float32))


zeros = np.zeros(SAMPLE_FRAMES).astype(np.float32)
ignored = (
    zeros,
    np.stack((zeros, zeros)),
)


def decode_audio(clean_file, noise_file):
    signal, _ = librosa.load(clean_file, sr=SAMPLE_RATE, dtype="float32")
    # signal, _ = sf.read(clean_file, dtype="float32")
    if signal.shape[0] <= SAMPLE_FRAMES:
        return ignored

    beg = np.random.randint(signal.shape[0] - SAMPLE_FRAMES)
    end = beg + SAMPLE_FRAMES
    clean_signal = signal[beg:end]

    signal, _ = librosa.load(noise_file, sr=SAMPLE_RATE, dtype="float32")
    # signal, _ = sf.read(noise_file, dtype="float32")
    if signal.shape[0] <= SAMPLE_FRAMES:
        return ignored

    beg = np.random.randint(signal.shape[0] - SAMPLE_FRAMES)
    end = beg + SAMPLE_FRAMES
    noise_signal = signal[beg:end]

    noisy = (
        clean_signal + np.sqrt(norm(clean_signal) / norm(noise_signal)) * noise_signal
    )
    return noisy, np.stack((clean_signal, noise_signal))


def data_generator(mode):
    clean_files, noise_files = data[mode]["clean"], data[mode]["noise"]
    while True:
        yield np.random.choice(clean_files), np.random.choice(noise_files)


if __name__ == "__main__":
    dataset = iter(get_dataset("train"))
    print(next(dataset))
