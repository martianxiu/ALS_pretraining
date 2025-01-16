# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.

import os
from os.path import join, exists
import pickle
import copy
import numpy as np
from scipy.stats import mode
import torch
import multiprocessing
import SharedArray
import torch.distributed as dist
from tqdm import tqdm
from pathlib import Path
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils
from ..dataset_als_bevmae_template import DatasetTemplate as DatasetTemplateALSBEVMAE
import laspy



name_id_mapping = {
    'S1': 0,
    'S2': 1,
    'S3': 2,
    'S4': 3,
    'S5': 4,
    'S6': 5,
    'S7': 6,
    'S8': 7,
    'S9': 8,
}


class OpenGFClsDataset(DatasetTemplateALSBEVMAE):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.loop = self.dataset_cfg.LOOP if self.mode != 'test' else 1
        self.data_path = self.root_path / self.dataset_cfg.PROCESSED_DATA_TAG[self.split]
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt') # connect paths = os.path.join(...)
        self.region_list = [x.strip() for x in open(split_dir).readlines()]
        self.ignore_index = self.dataset_cfg.IGNORE_INDEX
        self.info_limit = self.dataset_cfg.INFO_LIMIT
        self.pretrain_mode = self.dataset_cfg.PRETRAIN_MODE

        self.infos = []
        # self.labels = []
        self.include_als_data(self.mode) # load info file (not poitn cloud)

        self.use_shared_memory = self.dataset_cfg.get('USE_SHARED_MEMORY', False) and self.training != 'test'
        self.remaining_infos = []

        if self.use_shared_memory:
            self.shared_memory_file_limit = self.dataset_cfg.get('SHARED_MEMORY_FILE_LIMIT', 0x7FFFFFFF)
            self.load_data_to_shared_memory()
        else:
            self.shared_memory_file_limit = 0

        self.use_local_storage = self.dataset_cfg.get('USE_LOCAL_STORAGE', False) and self.training
        if self.use_local_storage:
            self.local_storage_path = os.getenv(self.dataset_cfg.get('LOCAL_STORAGE_PATH', None))
            self.local_storage_file_limit = self.dataset_cfg.get('LOCAL_STORAGE_FILE_LIMIT', 0x7FFFFFFF)
            if common_utils.is_main_process() and not exists(join(self.local_storage_path, "als")):
                os.makedirs(join(self.local_storage_path, "als"))
            self.load_data_to_local_storage()

    def include_als_data(self, mode):
        self.logger.info('Loading OpenGF Classification dataset')
        als_infos = []
        num_skipped_infos = 0
        for k in range(len(self.region_list)):
            region_name = os.path.splitext(self.region_list[k])[0]
            info_path = self.data_path / (f"{region_name}.pkl")
            print(info_path)
            if not info_path.exists():
                num_skipped_infos += 1
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                als_infos.extend(infos)

            

        n_original = len(als_infos)
        # limit info 
        if self.info_limit is not None and self.mode == 'train':
            n = np.ceil(len(als_infos) * self.info_limit).astype(int)
            als_infos = als_infos[:n]
            self.logger.info(f"info limit: {self.info_limit}. Use {n}/{n_original}")

        self.infos.extend(als_infos[:])
        self.logger.info('Total skipped info %s' % num_skipped_infos)
        self.logger.info('Total samples for PureForest dataset: %d' % (len(als_infos)))
        

        


    def load_data_to_shared_memory(self):
        self.logger.info(f'Loading {self.mode} data to shared memory (file limit={self.shared_memory_file_limit})')

        cur_rank, num_gpus = common_utils.get_dist_info()
        all_infos = self.infos[:self.shared_memory_file_limit] \
            if self.shared_memory_file_limit < len(self.infos) else self.infos
        cur_infos = all_infos[cur_rank::num_gpus] # split evenly for all GPUs
        

        for info in cur_infos:
            pc_info = info['point_cloud']
            sequence_name = pc_info['region_name']
            sample_idx = pc_info['sample_idx']

            sa_key = f'{sequence_name}___{sample_idx}'
            if os.path.exists(f"/dev/shm/{sa_key}"):
                continue

            points = self.get_lidar(sequence_name, sample_idx)
            common_utils.sa_create(f"shm://{sa_key}", points)
            self.logger.info(f"Saved {sa_key} ({points.nbytes/1024/1024:.4f} MB) to the shared memory.")

        if dist.is_initialized():
            dist.barrier()
        self.logger.info('Training data has been saved to shared memory')
        

    def load_data_to_local_storage(self):
        self.logger.info(f'Loading training data to local storage (file limit={self.local_storage_file_limit})')
        cur_rank, num_gpus = common_utils.get_dist_info()
        if self.shared_memory_file_limit != 0:
            self.logger.info("Will save in addition to the shared memory")
            start = self.shared_memory_file_limit
            end = start + self.local_storage_file_limit
            all_infos = self.infos[start:end] if end < len(self.infos) else self.infos
        else:
            self.logger.info("Will save to the local storage from the beginning")
            all_infos = self.infos[:self.local_storage_file_limit] if self.local_storage_file_limit < len(self.infos) else self.infos
        cur_infos = all_infos[cur_rank::num_gpus]
        for info in cur_infos:
            pc_info = info['point_cloud']
            sequence_name = pc_info['region_name']
            sample_idx = pc_info['sample_idx']

            filename = f'{sequence_name}___{sample_idx}.npy'
            if os.path.exists(join(self.local_storage_path, "als", filename)):
                continue

            points = self.get_lidar(sequence_name, sample_idx)
            np.save(join(self.local_storage_path, "als", filename), points)

        dist.barrier()
        self.logger.info('Training data has been saved to local storage')

    def clean_shared_memory(self):
        self.logger.info(f'Clean training data from shared memory (file limit={self.shared_memory_file_limit})')

        cur_rank, num_gpus = common_utils.get_dist_info()
        all_infos = self.infos[:self.shared_memory_file_limit] \
            if self.shared_memory_file_limit < len(self.infos) else self.infos
        cur_infos = all_infos[cur_rank::num_gpus]
        for info in cur_infos:
            pc_info = info['point_cloud']
            sequence_name = pc_info['region_name']
            sample_idx = pc_info['sample_idx']

            sa_key = f'{sequence_name}___{sample_idx}'
            if not os.path.exists(f"/dev/shm/{sa_key}"):
                continue

            SharedArray.delete(f"shm://{sa_key}")

        if num_gpus > 1:
            dist.barrier()
        self.logger.info('Training data has been deleted from shared memory')

    @staticmethod
    def check_sequence_name_with_all_version(sequence_file):
        if not sequence_file.exists():
            found_sequence_file = sequence_file
            for pre_text in ['training', 'validation', 'testing']:
                if not sequence_file.exists():
                    temp_sequence_file = Path(str(sequence_file).replace('segment', pre_text + '_segment'))
                    if temp_sequence_file.exists():
                        found_sequence_file = temp_sequence_file
                        break
            if not found_sequence_file.exists():
                found_sequence_file = Path(str(sequence_file).replace('_with_camera_labels', ''))
            if found_sequence_file.exists():
                sequence_file = found_sequence_file
        return sequence_file

    def get_infos(self, raw_data_path, save_path, num_workers=multiprocessing.cpu_count(), has_label=True, sampled_interval=1):
        from functools import partial
        from . import waymo_utils
        print('---------------The waymo sample interval is %d, total sequecnes is %d-----------------'
              % (sampled_interval, len(self.sample_sequence_list)))

        process_single_sequence = partial(
            waymo_utils.process_single_sequence,
            save_path=save_path, sampled_interval=sampled_interval, has_label=has_label
        )
        sample_sequence_file_list = [
            self.check_sequence_name_with_all_version(raw_data_path / sequence_file)
            for sequence_file in self.sample_sequence_list
        ]

        with multiprocessing.Pool(num_workers) as p:
            sequence_infos = list(tqdm(p.imap(process_single_sequence, sample_sequence_file_list),
                                       total=len(sample_sequence_file_list)))

        all_sequences_infos = [item for infos in sequence_infos for item in infos]
        return all_sequences_infos

    def get_lidar(self, region_name, sample_idx):
        def process_laz(laz_file_path):
            # Load the laz file
            with laspy.open(laz_file_path) as laz_file:
                point_data = laz_file.read()
            
            # Load xyz, intensity, return number, and number of returns
            xyz = np.vstack((point_data.x, point_data.y, point_data.z)).T
            intensity = point_data.intensity
            return_number = point_data.return_number
            num_returns = point_data.number_of_returns
            seg = np.array(point_data.classification) 
            
            valid_indices = np.logical_and(seg != 65, seg != 66) # discrard "artefact" and "synthetic"
            
            xyz = xyz[valid_indices]
            intensity = intensity[valid_indices][:, None] # num_point, 1
            return_number = return_number[valid_indices][:, None]
            num_returns = num_returns[valid_indices][:, None]
            return_ratio = return_number/num_returns
            seg = seg[valid_indices][:, None]

            # centralize x,y, and let z min = 0
            min_xyz = np.min(xyz, 0)
            max_xyz = np.max(xyz, 0)
            trans_xyz = (max_xyz - min_xyz) / 2
            trans_xyz[2] = 0 # zero z
            xyz = xyz - min_xyz - trans_xyz
            return np.concatenate([xyz, intensity, return_number, num_returns, return_ratio, seg], 1)

        lidar_file = self.data_path / f"{sample_idx}" 
        points_all = process_laz(lidar_file)  # (N, 8): [x, y, z, intensity, return_num, num_return, label]
        points_all[:, 3] = np.tanh(points_all[:, 3]) # normalize intensity by tanh? TODO
        return points_all

    def __len__(self):
        if self._merge_all_iters_to_one_epoch: # make all iterations into one epoch instead of multiple epochs 
            return len(self.infos) * self.total_epochs

        return len(self.infos) * self.loop

    def __getitem__(self, index):
        # if self._merge_all_iters_to_one_epoch:
        index = index % len(self.infos) # it applies because of the loop
        
        info = copy.deepcopy(self.infos[index])
        pc_info = info['point_cloud']
        region_name = pc_info['region_name']
        sample_idx = pc_info['sample_idx']

        if self.use_shared_memory and index < self.shared_memory_file_limit:
            sa_key = f'{region_name}___{sample_idx}'
            points = SharedArray.attach(f"shm://{sa_key}").copy()
        elif self.use_local_storage and index < self.shared_memory_file_limit + self.local_storage_file_limit:
            filename = f'{region_name}___{sample_idx}.npy'
            points = np.load(join(self.local_storage_path, "als", filename))
        else:
            points = self.get_lidar(region_name, sample_idx) # get lidar points 

        seg = points[:, -1:].astype(int)
        points = points[:, :-1]
        input_dict = {
            'points': points,
            'point_seg_labels_full': seg,
        }

        data_dict = self.prepare_data(data_dict=input_dict) # encode feature and process (ref. data processor)
        shuffle_idx = data_dict.pop('shuffle_idx', None) # because data processing invovles shuffling
        crop_idx = data_dict.pop('crop_indices', None)
        random_sample_idx = data_dict.pop('random_sample_indices', None)
        inverse_mapping = data_dict.get('inverse_mapping', None)
        
        minor_cls_name = sample_idx.split('_')[0] # Sx
        label_id = name_id_mapping[minor_cls_name]
        data_dict['labels'] = label_id

        data_dict['metadata'] = None
        data_dict.pop('num_points_in_gt', None)
        data_dict.pop('point_seg_labels_full', None)

        return data_dict

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            map_name_to_kitti = {
                'Vehicle': 'Car',
                'Pedestrian': 'Pedestrian',
                'Cyclist': 'Cyclist',
                'Sign': 'Sign',
                'Car': 'Car'
            }
            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        def waymo_eval(eval_det_annos, eval_gt_annos):
            from .waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
            eval = OpenPCDetWaymoDetectionMetricsEstimator()

            ap_dict = eval.waymo_evaluation(
                eval_det_annos, eval_gt_annos, class_name=class_names,
                distance_thresh=1000, fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            ap_result_str = '\n'
            for key in ap_dict:
                print(key)
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos)
        elif kwargs['eval_metric'] == 'waymo':
            ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict

    def create_groundtruth_database(self, info_path, save_path, used_classes=None, split='train', sampled_interval=10,
                                    processed_data_tag=None):
        database_save_path = save_path / ('%s_gt_database_%s_sampled_%d' % (processed_data_tag, split, sampled_interval))
        db_info_save_path = save_path / ('%s_waymo_dbinfos_%s_sampled_%d.pkl' % (processed_data_tag, split, sampled_interval))
        db_data_save_path = save_path / ('%s_gt_database_%s_sampled_%d_global.npy' % (processed_data_tag, split, sampled_interval))
        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        point_offset_cnt = 0
        stacked_gt_points = []
        for k in range(0, len(infos), sampled_interval):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]

            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']
            points = self.get_lidar(sequence_name, sample_idx)

            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            gt_boxes = annos['gt_boxes_lidar']

            if k % 4 != 0 and len(names) > 0:
                mask = (names == 'Vehicle')
                names = names[~mask]
                difficulty = difficulty[~mask]
                gt_boxes = gt_boxes[~mask]

            if k % 2 != 0 and len(names) > 0:
                mask = (names == 'Pedestrian')
                names = names[~mask]
                difficulty = difficulty[~mask]
                gt_boxes = gt_boxes[~mask]

            num_obj = gt_boxes.shape[0]
            if num_obj == 0:
                continue

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            for i in range(num_obj):
                filename = '%s_%04d_%s_%d.bin' % (sequence_name, sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]
                gt_points[:, :3] -= gt_boxes[i, :3]

                if (used_classes is None) or names[i] in used_classes:
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)

                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'sequence_name': sequence_name,
                               'sample_idx': sample_idx, 'gt_idx': i, 'box3d_lidar': gt_boxes[i],
                               'num_points_in_gt': gt_points.shape[0], 'difficulty': difficulty[i]}

                    # it will be used if you choose to use shared memory for gt sampling
                    stacked_gt_points.append(gt_points)
                    db_info['global_data_offset'] = [point_offset_cnt, point_offset_cnt + gt_points.shape[0]]
                    point_offset_cnt += gt_points.shape[0]

                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

        # it will be used if you choose to use shared memory for gt sampling
        stacked_gt_points = np.concatenate(stacked_gt_points, axis=0)
        np.save(db_data_save_path, stacked_gt_points)