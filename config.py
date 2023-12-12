#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2019/5/9 14:13
# @Author  : Eric Ching
from yacs.config import CfgNode as CN
import platform

_C = CN()
_C.DATASET = CN()

if "Win" in platform.system():
    _C.DATASET.DATA_ROOT = '/data1/zhn/macdata/all_data/2-MICCAI_BraTS_2018/MICCAI_BraTS_2018_Data_Training'
else:
    _C.DATASET.DATA_ROOT = "/data1/zhn/macdata/all_data/2-MICCAI_BraTS_2018/MICCAI_BraTS_2018_Data_Training"

_C.DATASET.NUM_FOLDS = 5
_C.DATASET.SELECT_FOLD = 0
_C.DATASET.USE_MODES = ("t1", "t2", "flair", "t1ce")
_C.DATASET.INPUT_SHAPE = (160, 192, 128)

_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = 1
_C.DATALOADER.NUM_WORKERS = 6

_C.MODEL = CN()
_C.MODEL.NAME = 'unet-vae'
_C.MODEL.INIT_CHANNELS = 16
_C.MODEL.DROPOUT = 0.2
_C.MODEL.LOSS_WEIGHT = 0.1

_C.SOLVER = CN()
_C.SOLVER.LEARNING_RATE = 1e-3
_C.SOLVER.WEIGHT_DECAY = 1e-5
_C.SOLVER.POWER = 0.9
_C.SOLVER.NUM_EPOCHS = 300

_C.MISC = CN()
_C.LOG_DIR = './logs'
