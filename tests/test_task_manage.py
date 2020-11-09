# -*- coding: utf-8 -*-
from unittest.mock import Mock

import pytest

import settings


def test_task_manager_processing(task_manager):
    to_return = [{'task_id': 0, 'frame_id': 1, 'path': 'tests/resources/images/tmp.jpg'}]
    task_manager.get_tasks = Mock(return_value=to_return)
    task_manager.set_tasks = Mock(return_value=None)
    task_manager.process_frames()


@pytest.mark.parametrize('frames', [[
    {
        'task_id': 0,
        'frame_id': 1,
        'path': 'tests/resources/images/tmp.jpg',
        'person_detection': [
            {'bbox': [0.36, 0.326, 0.496, 0.6], 'proba': 0.815},
            {'bbox': [0.517, 0.033, 0.594, 0.349], 'proba': 0.943}
        ]
    } for _ in range(settings.COUNT_FRAMES_TO_PROCESS)
]])
def test_task_manager_predict_frames_using_model(task_manager, frames):
    task_manager.storage_path = settings.ROOT
    processed_frames = task_manager.detect_pose(frames)
    expected_keys = {'frame_id', 'duration', 'result'}
    expected_results_keys = {'l_sho', 'r_sho', 'l_elb', 'r_elb', 'l_wri', 'r_wri'}
    assert len(processed_frames) == len(frames)
    for processed_task in processed_frames:
        assert expected_keys.intersection(processed_task.keys()) == expected_keys
        for result in processed_task['result']:
            assert expected_results_keys.intersection(result.keys()) == expected_results_keys
