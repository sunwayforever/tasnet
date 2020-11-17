#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2020-11-17 14:21
DATA_DIR = "/media/sunway/backup/tasnet_datasets/"

SAMPLE_RATE = 16000
SAMPLE_WINDOW_SIZE_MS = 1000
SAMPLE_FRAMES = SAMPLE_WINDOW_SIZE_MS * SAMPLE_RATE // 1000

N = 256
L = 20
B = 256
H = 512
P = 3
X = 8
R = 4
C = 2

BATCH_SIZE = 100
N_BATCH = 100
