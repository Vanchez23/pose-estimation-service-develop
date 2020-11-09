# -*- coding: utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSE_ESTIMATION_MODEL_RESOLUTION = (256, 192)
POSE_ESTIMATION_MODEL_FOLDER = os.path.join(ROOT, 'models', 'snapshot140')

STORAGE_PATH = os.path.join(ROOT, 'file_storage')

TASK_TIMEOUT = 5

VERSION = 'v0.2.0'
