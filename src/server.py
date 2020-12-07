# -*- coding: utf-8 -*-
import logging

from waitress import serve
from flask import Flask

import settings
from .background_worker import BackgroundWorker

logger = logging.getLogger(__name__)


class Server:

    def __init__(self):
        self.app = Flask(__name__)
        self.background_worker = None
        if settings.EXECUTOR_MODE:
            self.background_worker = BackgroundWorker()

    def run(self):
        if settings.EXECUTOR_MODE and self.background_worker is not None:
            logger.info('Background worker has been started.')
            self.background_worker.run()
        logger.info(f' Starting of service with settings variables: PORT: {settings.PORT},'
                    f' TESTING_MODE: {settings.TESTING_MODE},'
                    f' MODEL PATH: {settings.POSE_ESTIMATION_MODEL_FOLDER},'
                    f' MODEL RESOLUTION: {settings.POSE_ESTIMATION_MODEL_RESOLUTION},'
                    f' STORAGE PATH: {settings.STORAGE_PATH},'
                    f' STORAGE_SERVICE_API_URL: {settings.STORAGE_SERVICE_API_URL},'
                    f' DEVICE: {settings.DEVICE},'
                    f' COUNT_FRAMES_TO_PROCESS: {settings.COUNT_FRAMES_TO_PROCESS}')

        logger.info('Service has been started.')
        serve(self.app, host=settings.HOST, port=settings.PORT)

    def stop(self):
        if settings.EXECUTOR_MODE and self.background_worker is not None:
            self.background_worker.stop()
            logger.info('Background worker has been stopped.')
        logger.info('Service has been stopped.')
