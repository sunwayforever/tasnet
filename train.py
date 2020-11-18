#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2020-11-17 14:22
import tensorflow as tf
import numpy as np
import os
import argparse
import glob
import re

from tensorflow import keras
from tensorflow.keras import layers, losses, metrics, optimizers, models

from config import *
from models import *
from tasnet_dataset import *


def train():
    tf.keras.backend.clear_session()

    tasnet = TasNet()
    model = tasnet.model()
    model.summary()

    model.compile(
        optimizer=optimizers.Adam(),
        loss=TasNet.loss,
    )

    model.fit(
        data_generator("train"),
        steps_per_epoch=N_BATCH,
        batch_size=BATCH_SIZE,
        epochs=10,
        shuffle=False,
        verbose=1,
    )



if __name__ == "__main__":
    train()
