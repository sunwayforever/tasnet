#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2020-10-28 13:35
import io
import numpy as np
import tensorflow as tf
import soundfile as sf
from config import *
import pickle

with open(f"{DATA_DIR}/data.pkl", "rb") as f:
    data = pickle.load(f)

MIN_ENERGY = 1e-5


def data_generator(mode):
    clean_files, noise_files = data[mode]["clean"], data[mode]["noise"]

    def norm(x):
        return (x ** 2).mean()

    while True:
        mix, clean, noise = [], [], []
        for _ in range(BATCH_SIZE):
            index = np.random.choice(len(clean_files))
            signal, _ = sf.read(clean_files[index])
            if signal.shape[0] <= SAMPLE_FRAMES:
                continue
            beg = np.random.randint(signal.shape[0] - SAMPLE_FRAMES)
            end = beg + SAMPLE_FRAMES
            clean_signal = signal[beg:end]
            if norm(clean_signal) < MIN_ENERGY:
                continue

            index = np.random.choice(len(noise_files))
            signal, _ = sf.read(noise_files[index])
            if signal.shape[0] <= SAMPLE_FRAMES:
                continue
            beg = np.random.randint(signal.shape[0] - SAMPLE_FRAMES)
            end = beg + SAMPLE_FRAMES
            noise_signal = signal[beg:end]
            if norm(noise_signal) < MIN_ENERGY:
                continue
            clean.append(clean_signal)
            noise.append(noise_signal)

            mix = [a + b for a, b in zip(clean, noise)]

        yield (np.stack(mix), np.stack((np.stack(clean), np.stack(noise)), axis=1))


if __name__ == "__main__":
    mix, x = next(data_generator("train"))
    print(mix.shape)
    print(x.shape)
