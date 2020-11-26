# -*- coding: utf-8 -*-
import os
import pytest

import settings
from src.core.pose_estimation import HRNetModel
from settings.hrnet.config import cfg, update_config
from src.core import TaskManager

TEST_ROOT = os.path.abspath(os.path.dirname(__file__))


@pytest.yield_fixture(scope='session')
def pose_estimator():
    update_config(cfg, {'cfg': settings.POSE_ESTIMATION_CONFIG,
                        'modelDir': settings.POSE_ESTIMATION_MODEL_PATH})
    pe = HRNetModel(cfg,
                    settings.POSE_ESTIMATION_MODEL_RESOLUTION,
                    settings.DEVICE)
    yield pe


@pytest.yield_fixture(scope='module')
def task_manager(pose_estimator):
    tm = TaskManager(pose_estimation_model=pose_estimator, storage_path=settings.ROOT)
    yield tm
