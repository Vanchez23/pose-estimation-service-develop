# -*- coding: utf-8 -*-
import os
import pytest

import settings
from src.core import PoseEstimator
from src.core import TaskManager

TEST_ROOT = os.path.abspath(os.path.dirname(__file__))


@pytest.yield_fixture(scope='session')
def pose_estimator():
    pe = PoseEstimator(model_folder=settings.POSE_ESTIMATION_MODEL_FOLDER,
                       resolution=settings.POSE_ESTIMATION_MODEL_RESOLUTION)
    yield pe


@pytest.yield_fixture(scope='module')
def task_manager(pose_estimator):
    tm = TaskManager(pose_estimation_model=pose_estimator, storage_path=settings.ROOT)
    yield tm
