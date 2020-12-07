# -*- coding: utf-8 -*-
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSE_ESTIMATION_CONFIG = os.path.join(ROOT, 'settings/hrnet/w32_256x192_adam_lr1e-3.yaml')
POSE_ESTIMATION_MODEL_PATH = os.path.join(ROOT, 'models/hrnet/pose_hrnet_w32_256x192.pth')
POSE_ESTIMATION_MODEL_RESOLUTION = (256, 192)

STORAGE_PATH = os.path.join(ROOT, 'file_storage')

TASK_TIMEOUT = 5

VERSION = 'v0.3.0'
