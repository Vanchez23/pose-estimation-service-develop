# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math

import tensorflow as tf
from tfflat.base import Tester
from nms.nms import oks_nms
from utils.config import cfg
from utils.model import Model
from .constants import KEYPOINTS_NAMES

import settings

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def get_session():
    config = tf.ConfigProto()
    if settings.GPU_MODE:
        config.gpu_options.per_process_gpu_memory_fraction = settings.GPU_MEM_SPACE
        config.gpu_options.allow_growth = True
    return tf.Session(config=config)


class PoseEstimator:
    """
    Магический клас от TF-SimpleHumanPose в котором можно найти большое количество констант.
    """

    def __init__(self, model_folder, resolution=(256, 192)):
        """
        Params:
            resolution - network input resolution. default=(256,192)
                          Recommends :  (256,192), (384,288)
            model_name - 140
        Result:
            TfPoseEstimator
        """
        get_session().as_default()
        cfg.model_dump_dir = model_folder
        tester = Tester(Model(), cfg)
        tester.load_weights(140)  # 140 - по этому числу загружается снепшот модели
        # При загрузке по названию падает в ошибку при предсказании
        self.model = tester

        cfg.input_shape = resolution
        self.height, self.width = resolution

    def _get_dir(self, src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

    def _get_3rd_point(self, a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    def _get_affine_transform(self, center,
                              scale,
                              rot,
                              output_size,
                              shift=np.array([0, 0], dtype=np.float32),
                              inv=0):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            print(scale)
            scale = np.array([scale, scale])

        src_w = scale[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = self._get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale * shift
        src[1, :] = center + src_dir + scale * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2:, :] = self._get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = self._get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    def _postprocessing(self, kps_result, area_save):
        score_result = np.copy(kps_result[:, :, 2])
        score_result_keypoints = score_result.copy()

        kps_result[:, :, 2] = 1
        kps_result = kps_result.reshape(-1, cfg.num_kps * 3)

        rescored_score = np.zeros((len(score_result)))
        for i in range(len(score_result)):
            score_mask = score_result[i] > cfg.score_thr
            if np.sum(score_mask) > 0:
                rescored_score[i] = np.mean(score_result[i][score_mask])

        score_result = rescored_score
        keep = oks_nms(kps_result, score_result, area_save, cfg.oks_nms_thr)
        if len(keep) > 0:
            kps_result = kps_result[keep, :]
            score_result = score_result[keep]
            area_save = area_save[keep]

        return kps_result, score_result_keypoints, area_save

    def crop_image(self, image, bbox):
        """
        Вырезает прямоугольник из картинки
        с использованием афинных трансформаций.
        Взят у авторов сети.
        """
        img = image.copy()

        x, y = bbox[0], bbox[1]
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        aspect_ratio = cfg.input_shape[1] / cfg.input_shape[0]
        center = np.array([x + w * 0.5, y + h * 0.5])
        if w > aspect_ratio * h:
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array([w, h]) * 1.25
        rotation = 0

        trans = self._get_affine_transform(center, scale, rotation, (cfg.input_shape[1], cfg.input_shape[0]))
        cropped_img = cv2.warpAffine(img, trans, (cfg.input_shape[1], cfg.input_shape[0]), flags=cv2.INTER_LINEAR)

        cropped_img = cfg.normalize_input(cropped_img)

        crop_info = np.asarray([center[0] - scale[0] * 0.5, center[1] - scale[1] * 0.5, center[0] + scale[0] * 0.5,
                                center[1] + scale[1] * 0.5])

        return [cropped_img, crop_info]

    def filter_hands(self, keypoints, score, ndigits=3):
        """
        Метод приводит данные, предсказанные моделью,
        к необходимой структуре.
        Индексы начинаются с 5 по 11, так как в версии СОСО присутствуют
        другие точки, которые нам не нужны (Полный список точек
        можно посмотреть в переменной self.kps_names).

        Индексы в keypoints умножаются на 3, так как координаты:
        (x, y, v) для одной точки идут подряд. Переменная v
        используется для предстказания видимости точки (нам не нужна).
        """
        # keypoints = keypoints[5 * 3:11 * 3]
        # score = score[5:11]
        # names = KEYPOINTS_NAMES[5:11]
        names = KEYPOINTS_NAMES
        result = {}
        # pts_count = 6
        pts_count = 17
        for i, name in enumerate(names):
            x = float(round(keypoints[i * 3] / self.width, ndigits))
            y = float(round(keypoints[i * 3 + 1] / self.height, ndigits))

            if keypoints[i * 3] == 0 and keypoints[i * 3 + 1] == 0 and keypoints[i * 3 + 2] == 0:
                x, y = None, None
                pts_count -= 1

            # result[name[:5]] = {'x': x,
            #                     'y': y,
            #                     'proba': float(round(score[i], ndigits))}
            result[name] = {'x': x,
                                'y': y,
                                'proba': float(round(score[i], ndigits))}

        return result if pts_count >= 3 else []

    def predict(self, image, bboxes, ndigits=3):
        """
        Основной метод для предсказания ключевых точек.
        Взят у авторов сети.
        """
        answers = []

        self.height, self.width, _ = image.shape

        kps_result = np.zeros((len(bboxes), cfg.num_kps, 3))
        area_save = np.zeros(len(bboxes))

        for batch_id in range(0, len(bboxes), cfg.test_batch_size):
            start_id = batch_id
            end_id = min(len(bboxes), batch_id + cfg.test_batch_size)

            imgs = []
            crop_infos = []
            for i in range(start_id, end_id):
                bbox = np.array(bboxes[i]).astype(np.float32)
                img, crop_info = self.crop_image(image, bbox)

                imgs.append(img)
                crop_infos.append(crop_info)
            imgs = np.array(imgs)
            crop_infos = np.array(crop_infos)

            heatmap = self.model.predict_one([imgs])[0]
            if cfg.flip_test:
                flip_imgs = imgs[:, :, ::-1, :]
                flip_heatmap = self.model.predict_one([flip_imgs])[0]

                flip_heatmap = flip_heatmap[:, :, ::-1, :]
                for (q, w) in cfg.kps_symmetry:
                    flip_heatmap_w, flip_heatmap_q = flip_heatmap[:, :, :, w].copy(), flip_heatmap[:, :, :, q].copy()
                    flip_heatmap[:, :, :, q], flip_heatmap[:, :, :, w] = flip_heatmap_w, flip_heatmap_q
                flip_heatmap[:, :, 1:, :] = flip_heatmap.copy()[:, :, 0:-1, :]
                heatmap += flip_heatmap
                heatmap /= 2

            for image_id in range(start_id, end_id):
                for j in range(cfg.num_kps):
                    hm_j = heatmap[image_id - start_id, :, :, j]
                    idx = hm_j.argmax()
                    y, x = np.unravel_index(idx, hm_j.shape)

                    px = int(math.floor(x + 0.5))
                    py = int(math.floor(y + 0.5))
                    if 1 < px < cfg.output_shape[1] - 1 and 1 < py < cfg.output_shape[0] - 1:
                        diff = np.array([hm_j[py][px + 1] - hm_j[py][px - 1],
                                         hm_j[py + 1][px] - hm_j[py - 1][px]])
                        diff = np.sign(diff)
                        x += diff[0] * .25
                        y += diff[1] * .25
                    kps_result[image_id, j, :2] = (
                        x * cfg.input_shape[1] / cfg.output_shape[1], y * cfg.input_shape[0] / cfg.output_shape[0])
                    kps_result[image_id, j, 2] = hm_j.max() / 255

                    # map back to original images
                for j in range(cfg.num_kps):
                    kps_result[image_id, j, 0] = kps_result[image_id, j, 0] / cfg.input_shape[1] * (
                        crop_infos[image_id - start_id][2] - crop_infos[image_id - start_id][0]) + \
                        crop_infos[image_id - start_id][0]
                    kps_result[image_id, j, 1] = kps_result[image_id, j, 1] / cfg.input_shape[0] * (
                        crop_infos[image_id - start_id][3] - crop_infos[image_id - start_id][1]) + \
                        crop_infos[image_id - start_id][1]

                area_save[image_id] = (crop_infos[image_id - start_id][2] - crop_infos[image_id - start_id][0]) * (
                    crop_infos[image_id - start_id][3] - crop_infos[image_id - start_id][1])

        kps_result, score_result_keypoints, area_save = self._postprocessing(kps_result, area_save)

        for i in range(len(kps_result)):
            result = self.filter_hands(kps_result[i], score_result_keypoints[i], ndigits)
            answers.append(result)
            # answers.append(kps_result[i])

        return answers
