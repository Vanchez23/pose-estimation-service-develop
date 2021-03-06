# -*- coding: utf-8 -*-
import threading
import time
import logging

import settings
from .core.pose_estimation import PoseEstimator
from .core.task_manage import TaskManager

logger = logging.getLogger(__name__)


class BackgroundWorker:

    def __init__(self):
        self.is_running = False
        self.thread = None
        self.model = PoseEstimator(settings.POSE_ESTIMATION_MODEL_FOLDER,
                                   settings.POSE_ESTIMATION_MODEL_RESOLUTION)
        self.task_manager = TaskManager(self.model, storage_path=settings.STORAGE_PATH)

    def run(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(self.background_task())
            self.thread.start()

    def stop(self):
        if self.is_running and self.thread:
            self.is_running = False
            self.thread.join()

    def background_task(self):
        while self.is_running:
            logger.debug('*Start processing...')
            need_sleep = True
            try:
                need_sleep = self.task_manager.process_frames()
            except Exception as e:
                logger.exception(f'Произошла ошибка {e}, {type(e)}.')
            if need_sleep:
                time.sleep(settings.TASK_TIMEOUT)
            logger.debug('*Stop processing.')
