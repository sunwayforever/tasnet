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
from tensorflow.keras import layers, losses, metrics, optimizers, models, callbacks

from config import *
from models import *
from tasnet_dataset import *


parser = argparse.ArgumentParser()
parser.add_argument(
    "--epoch",
    type=int,
    default=1,
)

parser.add_argument(
    "--reset",
    action="store_true",
)
FLAGS = parser.parse_args()

model = None

tensorboard_callback = callbacks.TensorBoard(log_dir=f"/tmp/tasnet/{TASNET}")
save_model_callback = callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: epoch % 10 == 9 and model.save(SAVED_MODEL_PATH)
)


def train():
    global model
    tf.keras.backend.clear_session()

    tasnet = TasNet()

    if os.path.exists(SAVED_MODEL_PATH) and not FLAGS.reset:
        print("load model")
        model = keras.models.load_model(
            SAVED_MODEL_PATH, custom_objects={"loss": TasNet.loss}
        )
    else:
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
        epochs=FLAGS.epoch,
        callbacks=[save_model_callback, tensorboard_callback],
        validation_data=data_generator("test"),
        validation_freq=5,
        validation_steps=2,
        shuffle=False,
        verbose=1,
    )


if __name__ == "__main__":
    train()
