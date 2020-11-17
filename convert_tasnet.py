#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2020-10-27 10:24
import numpy as np
import os
import shutil
import random
import soundfile as sf
import pickle

# from auditok import split
from config import *
from collections import defaultdict
import re

os.chdir(DATA_DIR)

labels = {"test": {}, "train": {}}
all_data = defaultdict(list)

data = {}
for mode in ["test", "train"]:
    data[mode] = {}
    for cat in ["clean", "noise"]:
        data[mode][cat] = []
        for _, _, files in os.walk(f"{mode}/{cat}/", followlinks=True):
            for f in files:
                data[mode][cat].append(f"{DATA_DIR}/{mode}/{cat}/{f}")

with open(f"{DATA_DIR}/data.pkl", "wb") as f:
    pickle.dump(data, f)
