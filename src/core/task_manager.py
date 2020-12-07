# -*- coding: utf-8 -*-
import os
import time
import logging

from json_rpc_proxy import JsonRPCProxy, JsonRPCProxyError

from .utils import load_image
import settings

logger = logging.getLogger(__name__)


class TaskManager:

    def __init__(self, pose_estimation_model, storage_path):
        self.model = pose_estimation_model
        self.storage_api = JsonRPCProxy(settings.STORAGE_SERVICE_API_URL)
        self.storage_path = storage_path

    def get_frames(self):
        """
        Получает frames, на которых необходимо провести определение ключевых точек людей
        это либо кадры, которые не были проанализированы системой,
        либо кадры, которые были отправлены на анализ, но прошло много времени с момента отправки.
        Кадры должны пройти предварительный preprocessing обработку детекции bounding box людей.
        :param count: опциональный параметр, по умолчанию равен 50
        :return: результат в формате словаря
        [{'task_id': <task_id>, 'frame_id': <frame_id>, 'path': <path>,
        'person_detection': <json format for person detection result>}, ...]
        """
        try:
            frames = self.storage_api.get_frames_to_pose_estimation(count=settings.COUNT_FRAMES_TO_PROCESS)
        except JsonRPCProxyError as e:
            logger.error(f'Произошла ошибка при получении списка фреймов из сервиса хранения: {e}, {type(e)}.')
            return []
        return frames

    def set_frames(self, frames):
        """
        Проводит запись результатов определения ключевых точек в кадре
        :param frames: список словарей, где каждый словарь задан ввиде:
        {'frame_id': <frame_id>, 'duration': <duration>, 'results': results}
            frame_id: обязательный параметр, id изображения в таблице Frame
            duration: обязательный параметр, float
            result: обязательный параметр, json
        :return: {'success': True}
        """
        try:
            self.storage_api.set_pose_estimation_results(frames=frames)
        except JsonRPCProxyError as e:
            explanation = f'Произошла ошибка при отправке списка фреймов в сервис хранения: {e}, {type(e)}.'
            logger.error(explanation)
            return False
        return True

    def process_frames(self):
        need_sleep_time = True
        frames = self.get_frames()
        if not frames:
            return need_sleep_time

        answers = self.detect_pose(frames)
        sent = self.set_frames(answers)

        if sent:
            need_sleep_time = False

        return need_sleep_time

    def detect_pose(self, frames, ndigits=3):
        """
        Обрабатывает список файлов из хранилища.
        """
        data_to_send = []
        for frame in frames:
            try:
                logger.debug(f'**Frame is: {frame}')
                task_id = frame['task_id']
                frame_id = frame['frame_id']

                logging.info(f'Начала обработки кардра. task_id = {task_id}, frame_id = {frame_id}')

                image_path = os.path.join(self.storage_path, frame['path'])

                image = load_image(image_path)
                list_bboxes = self._get_bboxes(frame, image.shape)

                t = time.time()

                detections = self.model.predict_bboxes(image, list_bboxes, ndigits)

                duration = time.time() - t

                frame_info_to_send = {
                    'frame_id': frame_id,
                    'duration': round(float(duration), ndigits),
                    'result': detections
                }

                logging.info(f'Обработан кадр task_id {task_id}, frame_id {frame_id}.')
                logging.debug(f'Обработан кадр. task_id = {task_id}, frame_id = {frame_id}.'
                              f'Результат: {frame_info_to_send}')

                data_to_send.append(frame_info_to_send)
            except Exception as e:
                explanation = f'Произошла ошибка при обработке кадра {frame_id}, задания {task_id}: {e}, {type(e)}.'
                logger.error(explanation)
                self.storage_api.log_error(task_id=task_id, explanation=explanation[:1024])

            logging.info(f'Обработана порция кадров в количестве: {len(data_to_send)}.')
            logging.debug(f'Обработана порция кадров в количестве: {len(data_to_send)}. {data_to_send}')
        return data_to_send

    def _get_bboxes(self, frame, image_shape):
        h = image_shape[0]
        w = image_shape[1]

        bboxes = []
        for person in frame['person_detection']:
            bbox = person['bbox']
            bbox = [bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h]
            bboxes.append(bbox)
        return bboxes
