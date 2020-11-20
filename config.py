#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2020-11-17 14:21
import os

DATA_DIR = "/media/sunway/backup/tasnet_datasets/"

SAMPLE_RATE = 16000
SAMPLE_WINDOW_SIZE_MS = 1000
SAMPLE_FRAMES = SAMPLE_WINDOW_SIZE_MS * SAMPLE_RATE // 1000

N = 128  # encoder output
L = 20  # encoder kernel
B = 128  # bottleneck otuput
H = 128  # conv output
P = 3  # conv kernel
X = 8  # # of repeat
R = 4  # repeat
C = 2
K = (SAMPLE_FRAMES - L) * 2 // L + 1

BATCH_SIZE = 2
N_BATCH = 100

TASNET = os.environ.get("TASNET", "default")

MODEL_PATH = "./model/" + TASNET + "/"
SAVED_MODEL_PATH = MODEL_PATH + "/tasnet/"
