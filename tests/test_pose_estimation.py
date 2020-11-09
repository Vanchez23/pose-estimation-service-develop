# -*- coding: utf-8 -*-
import os

import pytest

from .conftest import TEST_ROOT
from src.core.utils import load_image


@pytest.mark.parametrize('image_path, detection_count', [
    (os.path.join(TEST_ROOT, 'resources', 'images', 'tmp.jpg'), [17, 17])])
def test_retina_tool_detection(image_path, detection_count, pose_estimator):
    bboxes = [[461.07, 234.84, 634.97, 432.24], [661.55, 23.74, 759.93, 250.94]]
    image = load_image(image_path)
    detections = pose_estimator.predict(image, bboxes)
    for i, detection in enumerate(detections):
        assert len(detection) == detection_count[i]
