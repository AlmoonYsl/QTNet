import mmcv
import numpy as np
import pyquaternion
import tempfile
from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp

from mmdet.datasets import DATASETS

from . import NuScenesDataset

@DATASETS.register_module()
class NuScenesDatasetMemory(NuScenesDataset):
    def __init__(self,
                 num_frames=0,
                 features_root=None,
                 pred_result_root=None,
                 **kwargs):
        self.num_frames = num_frames
        self.features_root = features_root
        self.pred_result_root = pred_result_root

        self.multi_frames_enable = num_frames >= 1
        if self.multi_frames_enable:
            self.token_index_map = dict()

        super().__init__(**kwargs)

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        if self.multi_frames_enable:
            for i in range(len(data_infos)):
                self.token_index_map[data_infos[i]['token']] = i
        return data_infos

    def get_data_info(self, index):
        info = self.data_infos[index]

        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
            prev_sample_idx=info['prev'],
        )

        if self.multi_frames_enable:
            assert self.features_root is not None
            from nuscenes.utils.geometry_utils import transform_matrix
            from pyquaternion import Quaternion
            if self.modality['use_lidar']:
                modal_path = info['lidar_path']
            else:
                modal_path = info['cams']['CAM_FRONT']['data_path']
            features_filename = self.features_root + modal_path.split('/')[-1].split('.')[0] + '.bin.npy'
            pred_results_filename = self.pred_result_root + modal_path.split('/')[-1].split('.')[0] + '.bin'
            lidar2ego = transform_matrix(
                info['lidar2ego_translation'], Quaternion(info['lidar2ego_rotation'])
            )
            ego2global = transform_matrix(
                info['ego2global_translation'], Quaternion(info['ego2global_rotation'])
            )
            poses = dict(lidar2ego=lidar2ego.astype(np.float32),
                         ego2global=ego2global.astype(np.float32))
            input_dict['features_filename'] = features_filename
            input_dict['pred_results_filename'] = pred_results_filename
            input_dict['poses'] = poses

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        from mmcv.parallel.data_container import DataContainer
        if self.test_mode:
            data = self.prepare_test_data(idx)
            if self.multi_frames_enable:
                for i in range(len(data['img_metas'])):
                    data_list_dict = dict()
                    keys = list(data.keys())
                    for key in keys:
                        data_list_dict[key] = [
                            data[key][i].data if isinstance(data[key][i], DataContainer) else data[key][i]]
                    cur_data = data
                    for _ in range(self.num_frames - 1):
                        prev_token = cur_data['img_metas'][i].data['prev_sample_idx']
                        if prev_token == '':
                            prev_token = cur_data['img_metas'][i].data['sample_idx']
                        prev_data = self.prepare_test_data(self.token_index_map[prev_token])
                        if prev_data is None:
                            prev_data = cur_data
                        for key in keys:
                            data_list_dict[key].append(
                                prev_data[key][i].data if isinstance(prev_data[key][i], DataContainer) else
                                prev_data[key][i])
                        cur_data = prev_data
                    for key in keys:
                        if key == 'queries':
                            data[key][i] = np.stack(data_list_dict[key], axis=0)
                        elif key == 'pred_results':
                            for key2 in ['boxes', 'scores', 'labels']:
                                data[key][i][key2] = np.stack([data_list[key2] for data_list in data_list_dict[key]],
                                                              axis=0)
                        else:
                            data['img_metas'][i].data['prev_' + key] = data_list_dict[key][1:]
            return data
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            if self.multi_frames_enable:
                data_list_dict = dict()
                keys = list(data.keys())
                for key in keys:
                    data_list_dict[key] = [data[key].data if isinstance(data[key], DataContainer) else data[key]]
                cur_data = data
                for _ in range(self.num_frames - 1):
                    prev_token = cur_data['img_metas'].data['prev_sample_idx']
                    if prev_token == '':
                        prev_token = cur_data['img_metas'].data['sample_idx']
                    prev_data = self.prepare_train_data(self.token_index_map[prev_token])
                    if prev_data is None:
                        prev_data = cur_data
                    for key in keys:
                        data_list_dict[key].append(
                            prev_data[key].data if isinstance(prev_data[key], DataContainer) else prev_data[key])
                    cur_data = prev_data
                for key in keys:
                    if key == 'queries':
                        data[key] = np.stack(data_list_dict[key], axis=0)
                    elif key == 'pred_results':
                        for key2 in ['boxes', 'scores', 'labels']:
                            data[key][key2] = np.stack([data_list[key2] for data_list in data_list_dict[key]], axis=0)
                    else:
                        data['img_metas'].data['prev_' + key] = data_list_dict[key][1:]
            return data
