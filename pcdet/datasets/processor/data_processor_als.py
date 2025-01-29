from functools import partial

import numpy as np
import sys
from skimage import transform

from ...utils import box_utils, common_utils
from sklearn.cluster import DBSCAN
import pyransac3d as pyrsc
from spconv.pytorch.utils import PointToVoxel
import torch

tv = None
try:
    import cumm.tensorview as tv
except:
    pass


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output # voxel feature, grid coordinates , and num point in each voxel cell
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        # self.mode = 'train' if training else 'test'
        self.mode = training if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self.voxel_generator = None

        self.processor_configs = processor_configs

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)


    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None): # remove points outside the range
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config) # when initialized just return the function 

        if data_dict.get('points', None) is not None:
            mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
            data_dict['points'] = data_dict['points'][mask]

        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points
            data_dict['shuffle_idx'] = shuffle_idx

        return data_dict

    def percentile_clip_z(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.percentile_clip_z, config=config)
        
        high_p = config.PERCENTILE_HIGH 
        low_p = config.PERCENTILE_LOW
        z = data_dict["points"][:, 2]

        high = np.percentile(z, high_p)
        low = np.percentile(z, low_p)
        mask = np.logical_and(z > low, z < high)
        data_dict["points"] = data_dict["points"][mask]
        # np.save("percentile_points.npy", data_dict["points"])
        # sys.exit()
        return data_dict

    def pillar_crop(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.pillar_crop, config=config)
        def grid_sampling_2d(xy, grid_size):
            def fnv_hash_vec(arr):
                """
                FNV64-1A
                """
                assert arr.ndim == 2
                # Floor first for negative coordinates
                arr = arr.copy()
                arr = arr.astype(np.uint64, copy=False)
                hashed_arr = np.uint64(14695981039346656037) * np.ones(
                    arr.shape[0], dtype=np.uint64
                )
                for j in range(arr.shape[1]):
                    hashed_arr *= np.uint64(1099511628211)
                    hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
                return hashed_arr

            scaled_coord = xy / grid_size
            grid_coord = np.floor(scaled_coord).astype(int)
            min_coord = grid_coord.min(0)
            grid_coord -= min_coord
            scaled_coord -= min_coord
            # min_coord = min_coord * grid_size
            key = fnv_hash_vec(grid_coord) # unique key for each voxel id
            idx_sort = np.argsort(key) # sort keys for efficient processing
            key_sort = key[idx_sort]
            _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            return xy[idx_unique] 
            
        def crop_point_cloud(points, center, length):
            x_center, y_center = center
            half_length = length / 2

            # Define the bounding box
            x_min = x_center - half_length
            x_max = x_center + half_length
            y_min = y_center - half_length
            y_max = y_center + half_length

            # Filter points within the bounding box using np.logical_and
            condition_x = np.logical_and(points[:, 0] >= x_min, points[:, 0] <= x_max)
            condition_y = np.logical_and(points[:, 1] >= y_min, points[:, 1] <= y_max)
            crop_indices = np.logical_and(condition_x, condition_y)
            cropped_points = points[crop_indices]

            return cropped_points, crop_indices

        pillar_size = config.PILLAR_SIZE
        grid_size = config.GRID_SIZE_2D
        points = data_dict['points']
        grid_points_2d = grid_sampling_2d(points[:, :2], grid_size)
        crop_center = grid_points_2d[np.random.choice(np.arange(len(grid_points_2d)))]
        cropped, crop_indices = crop_point_cloud(points, crop_center, pillar_size)
        cropped[:, :2] = cropped[:, :2] - crop_center # centralize x and y
        cropped[:, 2] = cropped[:, 2] - np.min(cropped[:, 2]) # let min height = 0
        data_dict['points'] = cropped
        data_dict['crop_indices'] = crop_indices
        return data_dict

    # def pillar_crop_with_pc_range_clip_check(self, data_dict=None, config=None):
    #     if data_dict is None:
    #         return partial(self.pillar_crop_with_pc_range_clip_check, config=config)
    #     def grid_sampling_2d(xy, grid_size):
    #         def fnv_hash_vec(arr):
    #             """
    #             FNV64-1A
    #             """
    #             assert arr.ndim == 2
    #             # Floor first for negative coordinates
    #             arr = arr.copy()
    #             arr = arr.astype(np.uint64, copy=False)
    #             hashed_arr = np.uint64(14695981039346656037) * np.ones(
    #                 arr.shape[0], dtype=np.uint64
    #             )
    #             for j in range(arr.shape[1]):
    #                 hashed_arr *= np.uint64(1099511628211)
    #                 hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    #             return hashed_arr

    #         scaled_coord = xy / grid_size
    #         grid_coord = np.floor(scaled_coord).astype(int)
    #         min_coord = grid_coord.min(0)
    #         grid_coord -= min_coord
    #         scaled_coord -= min_coord
    #         # min_coord = min_coord * grid_size
    #         key = fnv_hash_vec(grid_coord) # unique key for each voxel id
    #         idx_sort = np.argsort(key) # sort keys for efficient processing
    #         key_sort = key[idx_sort]
    #         _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
    #         idx_select = (
    #             np.cumsum(np.insert(count, 0, 0)[0:-1])
    #             + np.random.randint(0, count.max(), count.size) % count
    #         )
    #         idx_unique = idx_sort[idx_select]
    #         return xy[idx_unique] 
            
    #     def crop_point_cloud(points, center, length):
    #         x_center, y_center = center
    #         half_length = length / 2

    #         # Define the bounding box
    #         x_min = x_center - half_length
    #         x_max = x_center + half_length
    #         y_min = y_center - half_length
    #         y_max = y_center + half_length

    #         # Filter points within the bounding box using np.logical_and
    #         condition_x = np.logical_and(points[:, 0] >= x_min, points[:, 0] <= x_max)
    #         condition_y = np.logical_and(points[:, 1] >= y_min, points[:, 1] <= y_max)
    #         crop_indices = np.logical_and(condition_x, condition_y)
    #         cropped_points = points[crop_indices]

    #         return cropped_points, crop_indices

    #     pillar_size = config.PILLAR_SIZE
    #     grid_size = config.GRID_SIZE_2D
    #     point_cloud_range = config.POINT_CLOUD_RANGE
    #     min_point = config.MIN_POINT
    #     points = data_dict['points']
    #     while True:
    #         grid_points_2d = grid_sampling_2d(points[:, :2], grid_size)
    #         crop_center = grid_points_2d[np.random.choice(np.arange(len(grid_points_2d)))]
    #         cropped, crop_indices = crop_point_cloud(points, crop_center, pillar_size)
    #         cropped[:, :2] = cropped[:, :2] - crop_center # centralize x and y
    #         cropped[:, 2] = cropped[:, 2] - np.min(cropped[:, 2]) # let min height = 0
    #         mask = common_utils.mask_points_by_range(cropped, point_cloud_range)
    #         cropped = cropped[mask]   
    #         if len(cropped) > min_point:
    #             break
    #     data_dict['points'] = cropped
    #     data_dict['crop_indices'] = crop_indices[mask]
    #     return data_dict

    def pillar_crop_no_min_z(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.pillar_crop_no_min_z, config=config)
        def grid_sampling_2d(xy, grid_size):
            def fnv_hash_vec(arr):
                """
                FNV64-1A
                """
                assert arr.ndim == 2
                # Floor first for negative coordinates
                arr = arr.copy()
                arr = arr.astype(np.uint64, copy=False)
                hashed_arr = np.uint64(14695981039346656037) * np.ones(
                    arr.shape[0], dtype=np.uint64
                )
                for j in range(arr.shape[1]):
                    hashed_arr *= np.uint64(1099511628211)
                    hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
                return hashed_arr

            scaled_coord = xy / grid_size
            grid_coord = np.floor(scaled_coord).astype(int)
            min_coord = grid_coord.min(0)
            grid_coord -= min_coord
            scaled_coord -= min_coord
            # min_coord = min_coord * grid_size
            key = fnv_hash_vec(grid_coord) # unique key for each voxel id
            idx_sort = np.argsort(key) # sort keys for efficient processing
            key_sort = key[idx_sort]
            _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            return xy[idx_unique] 
            
        def crop_point_cloud(points, center, length):
            x_center, y_center = center
            half_length = length / 2

            # Define the bounding box
            x_min = x_center - half_length
            x_max = x_center + half_length
            y_min = y_center - half_length
            y_max = y_center + half_length

            # Filter points within the bounding box using np.logical_and
            condition_x = np.logical_and(points[:, 0] >= x_min, points[:, 0] <= x_max)
            condition_y = np.logical_and(points[:, 1] >= y_min, points[:, 1] <= y_max)
            crop_indices = np.logical_and(condition_x, condition_y)
            cropped_points = points[crop_indices]

            return cropped_points, crop_indices

        pillar_size = config.PILLAR_SIZE
        grid_size = config.GRID_SIZE_2D
        points = data_dict['points']
        grid_points_2d = grid_sampling_2d(points[:, :2], grid_size)
        crop_center = grid_points_2d[np.random.choice(np.arange(len(grid_points_2d)))]
        cropped, crop_indices = crop_point_cloud(points, crop_center, pillar_size)
        cropped[:, :2] = cropped[:, :2] - crop_center # centralize x and y
        # cropped[:, 2] = cropped[:, 2] - np.min(cropped[:, 2]) # let min height = 0
        data_dict['points'] = cropped
        data_dict['crop_indices'] = crop_indices
        return data_dict



    def pillar_crop_robust(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.pillar_crop_robust, config=config)
        def grid_sampling_2d(xy, grid_size):
            def fnv_hash_vec(arr):
                """
                FNV64-1A
                """
                assert arr.ndim == 2
                # Floor first for negative coordinates
                arr = arr.copy()
                arr = arr.astype(np.uint64, copy=False)
                hashed_arr = np.uint64(14695981039346656037) * np.ones(
                    arr.shape[0], dtype=np.uint64
                )
                for j in range(arr.shape[1]):
                    hashed_arr *= np.uint64(1099511628211)
                    hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
                return hashed_arr

            scaled_coord = xy / grid_size
            grid_coord = np.floor(scaled_coord).astype(int)
            min_coord = grid_coord.min(0)
            grid_coord -= min_coord
            scaled_coord -= min_coord
            # min_coord = min_coord * grid_size
            key = fnv_hash_vec(grid_coord) # unique key for each voxel id
            idx_sort = np.argsort(key) # sort keys for efficient processing
            key_sort = key[idx_sort]
            _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            return xy[idx_unique] 
            
        def crop_point_cloud(points, center, length):
            x_center, y_center = center
            half_length = length / 2

            # Define the bounding box
            x_min = x_center - half_length
            x_max = x_center + half_length
            y_min = y_center - half_length
            y_max = y_center + half_length

            # Filter points within the bounding box using np.logical_and
            condition_x = np.logical_and(points[:, 0] >= x_min, points[:, 0] <= x_max)
            condition_y = np.logical_and(points[:, 1] >= y_min, points[:, 1] <= y_max)
            crop_indices = np.logical_and(condition_x, condition_y)
            cropped_points = points[crop_indices]

            return cropped_points, crop_indices

        pillar_size = config.PILLAR_SIZE
        grid_size = config.GRID_SIZE_2D
        min_point = config.MIN_POINT
        points = data_dict['points']
        grid_points_2d = grid_sampling_2d(points[:, :2], grid_size)
        while True:
            crop_center = grid_points_2d[np.random.choice(np.arange(len(grid_points_2d)))]
            cropped, crop_indices = crop_point_cloud(points, crop_center, pillar_size)
            if len(cropped) < min_point:
                continue
            else:
                break
        cropped[:, :2] = cropped[:, :2] - crop_center # centralize 
        cropped[:, 2] = cropped[:, 2] - np.min(cropped[:, 2]) # let min height = 0
        data_dict['points'] = cropped
        data_dict['crop_indices'] = crop_indices
        return data_dict


    def pillar_crop_robust_v2(self, data_dict=None, config=None):
        # this is to reject points that two few points after voxelization 
        if data_dict is None:
            return partial(self.pillar_crop_robust_v2, config=config)
        def grid_sampling_2d(xy, grid_size):
            def fnv_hash_vec(arr):
                """
                FNV64-1A
                """
                assert arr.ndim == 2
                # Floor first for negative coordinates
                arr = arr.copy()
                arr = arr.astype(np.uint64, copy=False)
                hashed_arr = np.uint64(14695981039346656037) * np.ones(
                    arr.shape[0], dtype=np.uint64
                )
                for j in range(arr.shape[1]):
                    hashed_arr *= np.uint64(1099511628211)
                    hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
                return hashed_arr

            scaled_coord = xy / grid_size
            grid_coord = np.floor(scaled_coord).astype(int)
            min_coord = grid_coord.min(0)
            grid_coord -= min_coord
            scaled_coord -= min_coord
            # min_coord = min_coord * grid_size
            key = fnv_hash_vec(grid_coord) # unique key for each voxel id
            idx_sort = np.argsort(key) # sort keys for efficient processing
            key_sort = key[idx_sort]
            _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            return xy[idx_unique] 
            
        def crop_point_cloud(points, center, length):
            x_center, y_center = center
            half_length = length / 2

            # Define the bounding box
            x_min = x_center - half_length
            x_max = x_center + half_length
            y_min = y_center - half_length
            y_max = y_center + half_length

            # Filter points within the bounding box using np.logical_and
            condition_x = np.logical_and(points[:, 0] >= x_min, points[:, 0] <= x_max)
            condition_y = np.logical_and(points[:, 1] >= y_min, points[:, 1] <= y_max)
            cropped_points = points[np.logical_and(condition_x, condition_y)]

            return cropped_points

        pillar_size = config.PILLAR_SIZE
        grid_size = config.GRID_SIZE_2D
        grid_size_to_check = config.GRID_SIZE_TO_CHECK
        mask_ratio = config.MASK_RATIO
        min_point = int(1/(1-mask_ratio))
        
        points = data_dict['points']
        grid_points_2d = grid_sampling_2d(points[:, :2], grid_size)
        while True:
            crop_center = grid_points_2d[np.random.choice(np.arange(len(grid_points_2d)))]
            cropped = crop_point_cloud(points, crop_center, pillar_size)
            grid_points_to_check = grid_sampling_2d(cropped[:, :2], grid_size_to_check)
            if len(grid_points_to_check) < min_point:
                continue
            else:
                break
        cropped[:, :2] = cropped[:, :2] - crop_center # centralize 
        data_dict['points'] = cropped
        return data_dict



    def voxel_downsample(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE) # (max range - min range) / voxel size per axis
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE # [vx, vy, vz]
            return partial(self.voxel_downsample, config=config)

        def voxel_downsampling(points, grid_size):
            def fnv_hash_vec(arr):
                """
                FNV64-1A
                """
                assert arr.ndim == 2
                # Floor first for negative coordinates
                arr = arr.copy()
                arr = arr.astype(np.uint64, copy=False)
                hashed_arr = np.uint64(14695981039346656037) * np.ones(
                    arr.shape[0], dtype=np.uint64
                )
                for j in range(arr.shape[1]):
                    hashed_arr *= np.uint64(1099511628211)
                    hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
                return hashed_arr
            def ravel_hash_vec(arr):
                """
                Ravel the coordinates after subtracting the min coordinates.
                """
                assert arr.ndim == 2
                arr = arr.copy()
                arr -= arr.min(0)
                arr = arr.astype(np.uint64, copy=False)
                arr_max = arr.max(0).astype(np.uint64) + 1

                keys = np.zeros(arr.shape[0], dtype=np.uint64)
                # Fortran style indexing
                for j in range(arr.shape[1] - 1):
                    keys += arr[:, j]
                    keys *= arr_max[j + 1]
                keys += arr[:, -1]
                return keys

            xyz = points[:, :3]
            scaled_coord = xyz / grid_size
            grid_coord = np.floor(scaled_coord).astype(int)
            min_coord = grid_coord.min(0)
            grid_coord -= min_coord
            scaled_coord -= min_coord
            # min_coord = min_coord * grid_size
            key = fnv_hash_vec(grid_coord)
            idx_sort = np.argsort(key)
            key_sort = key[idx_sort]
            _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)

            # if mode == "train":  # train mode
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            points = points[idx_unique]
            
            return points, idx_unique, inverse

        voxel_size = config.VOXEL_SIZE # a list
        points = data_dict['points']
        down_sampled_points, idx_unique, inverse_index = voxel_downsampling(points, np.array(voxel_size))
        data_dict['points'] = down_sampled_points
        return data_dict

    def voxel_random_sampling(self, data_dict=None, config=None):
        if data_dict is None:
            # grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE) # (max range - min range) / voxel size per axis
            # self.grid_size = np.round(grid_size).astype(np.int64)
            # self.voxel_size = config.VOXEL_SIZE # [vx, vy, vz]
            return partial(self.voxel_downsample, config=config)

        def voxel_downsampling(points, grid_size):
            def fnv_hash_vec(arr):
                """
                FNV64-1A
                """
                assert arr.ndim == 2
                # Floor first for negative coordinates
                arr = arr.copy()
                arr = arr.astype(np.uint64, copy=False)
                hashed_arr = np.uint64(14695981039346656037) * np.ones(
                    arr.shape[0], dtype=np.uint64
                )
                for j in range(arr.shape[1]):
                    hashed_arr *= np.uint64(1099511628211)
                    hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
                return hashed_arr
            def ravel_hash_vec(arr):
                """
                Ravel the coordinates after subtracting the min coordinates.
                """
                assert arr.ndim == 2
                arr = arr.copy()
                arr -= arr.min(0)
                arr = arr.astype(np.uint64, copy=False)
                arr_max = arr.max(0).astype(np.uint64) + 1

                keys = np.zeros(arr.shape[0], dtype=np.uint64)
                # Fortran style indexing
                for j in range(arr.shape[1] - 1):
                    keys += arr[:, j]
                    keys *= arr_max[j + 1]
                keys += arr[:, -1]
                return keys

            xyz = points[:, :3]
            scaled_coord = xyz / grid_size
            grid_coord = np.floor(scaled_coord).astype(int)
            min_coord = grid_coord.min(0)
            grid_coord -= min_coord
            scaled_coord -= min_coord
            # min_coord = min_coord * grid_size
            key = fnv_hash_vec(grid_coord)
            idx_sort = np.argsort(key)
            key_sort = key[idx_sort]
            _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)

            # if mode == "train":  # train mode
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            points = points[idx_unique]
            return points, idx_unique, inverse

        voxel_size = config.VOXEL_SIZE # a list
        points = data_dict['points']
        # original_points = points
        points, idx_unique, inverse_index = voxel_downsampling(points, np.array(voxel_size))
        num_points = config.NUM_POINTS
        len_points = len(points)
        if num_points < len_points:
            choice = np.random.choice(np.arange(len(points)), num_points, replace=False)
        # elif num_points > len_points:
        #     len_original = len(original_points)
        #     choice = np.random.choice(np.arange(len(points)), num_points, replace=True)
        else:
            choice = np.arange(len(points))
        data_dict['points'] = points[choice]
        data_dict['voxel_sampled_indices_data_process'] = idx_unique
        data_dict['random_sampled_indices_data_process'] = choice
        return data_dict


    def transform_points_to_voxels_numpy(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE) # (max range - min range) / voxel size per axis
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE # [vx, vy, vz]
            return partial(self.transform_points_to_voxels_numpy, config=config)

        def voxel_downsampling(points, grid_size, max_pts_per_voxel, max_voxel, mode):
            def fnv_hash_vec(arr):
                """
                FNV64-1A
                """
                assert arr.ndim == 2
                # Floor first for negative coordinates
                arr = arr.copy()
                arr = arr.astype(np.uint64, copy=False)
                hashed_arr = np.uint64(14695981039346656037) * np.ones(
                    arr.shape[0], dtype=np.uint64
                )
                for j in range(arr.shape[1]):
                    hashed_arr *= np.uint64(1099511628211)
                    hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
                return hashed_arr

            xyz = points[:, :3]
            scaled_coord = xyz / grid_size
            grid_coord = np.floor(scaled_coord).astype(int)
            min_coord = grid_coord.min(0)
            grid_coord -= min_coord
            scaled_coord -= min_coord
            # min_coord = min_coord * grid_size
            key = fnv_hash_vec(grid_coord)
            idx_sort = np.argsort(key)
            key_sort = key[idx_sort]
            _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)

            # random sampling for max_pts_per_voxel_times 
            # sampled_point_list = [] # for voxels
            sampled_index_list = [] # for indexing labels 
            for i in range(max_pts_per_voxel):
                random_sample_index_residual = np.random.randint(0, count.max(), count.size) % count
                idx_select = (
                    np.cumsum(np.insert(count, 0, 0)[0:-1])
                    + random_sample_index_residual
                )
                idx_unique = idx_sort[idx_select] # unique and sorted indices
                # sampled_point_list.append(points[idx_unique][:, None, :])
                sampled_index_list.append(idx_unique[:, None])
            sampled_indices = np.concatenate(sampled_index_list, 1) # (n_voxel, max_pts_per_voxel) if train and (n_voxel, n * max_pts_per_voxel)
            voxels = points[sampled_indices] # n_voxel, max_pts_per_voxel, num_feature
            grid_coord = grid_coord[idx_unique] # voxel positions don't change by sampling. 
            
            # obtain inverse mapping: https://github.com/Pointcept/Pointcept/blob/7b37078ae301288309d62c4c88401716b6bdbe0e/pointcept/datasets/transform.py#L817
            inverse_mapping = np.zeros_like(inverse)
            inverse_mapping[idx_sort] = inverse
            
            # dont downsample voxels when test
            if max_voxel is not None and max_voxel < len(voxels) and mode != 'test':
                random_index = np.random.choice(np.arange(len(voxels)), max_voxel, replace=False)
                voxels = voxels[random_index]
                sampled_indices = sampled_indices[random_index]
                grid_coord = grid_coord[random_index]
            num_uniq_pts_per_voxel = np.apply_along_axis(lambda x: len(np.unique(x)), axis=1, arr=sampled_indices)
            
            return voxels, grid_coord, sampled_indices, num_uniq_pts_per_voxel, inverse_mapping

        voxel_size = config.VOXEL_SIZE # a list
        points = data_dict['points']
        
        voxels, grid_coord, sampled_indices, num_uniq_pts_per_voxel, inverse_mapping = voxel_downsampling(
            points=points, 
            grid_size=np.array(voxel_size),
            max_pts_per_voxel=config.MAX_POINTS_PER_VOXEL,
            max_voxel=config.MAX_NUMBER_OF_VOXELS[self.mode],
            mode=self.mode
        )
        
        grid_coord = grid_coord[:, ::-1]# reverse the order of dims
        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
        data_dict['voxels'] = voxels # should be features (N, max_pts_per_voxel, num_feature)
        data_dict['sampled_indices'] = sampled_indices # (N, max_pts_per_voxel)
        data_dict['voxel_coords'] = grid_coord # vox coordinates  (N, 3)
        data_dict['voxel_num_points'] = num_uniq_pts_per_voxel # num unique points in side a voxel
        data_dict['inverse_mapping'] = inverse_mapping # inverse mapping from the unique voxels to original points. it cannot be used if random samplign is enabled. 
        return data_dict


    def transform_points_to_voxels_placeholder(self, data_dict=None, config=None):
        # just calculate grid size
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels_placeholder, config=config)
        
        return data_dict
        
    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE) # (max range - min range) / voxel size per axis
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE # [vx, vy, vz]
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config) # return partial when initialization 

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE, # dataset voxel size. [0.1, 0.1, 0.15]
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points = data_dict['points']
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output # voxels: (n_voxel, max_n_point_per_voxel, C); coordinates: (n_voxel, 4[b_idx, z_idx, y_idx, x_idx])

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['voxels'] = voxels # should be features 
        data_dict['voxel_coords'] = coordinates # coordinates 
        data_dict['voxel_num_points'] = num_points # num points in side a voxel
        # print(voxels.shape)

        ##############################

        self.voxel_generator_gt = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE_BEV, # dataset voxel size for BEV: [0.8, 0.8, 6]
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL_BEV,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points_gt = data_dict['points']
        voxel_output_gt = self.voxel_generator_gt.generate(points_gt)
        voxels_gt, coordinates_gt, num_points_gt = voxel_output_gt

        data_dict['voxels_bev'] = voxels_gt
        data_dict['voxel_coords_bev'] = coordinates_gt
        data_dict['voxel_num_points_bev'] = num_points_gt
        # print(voxels_gt.shape, coordinates_gt.shape, num_points_gt.shape)
        # print(voxels_gt.shape)


        return data_dict

    def transform_points_to_voxels_no_bev(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE) # (max range - min range) / voxel size per axis
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE # [vx, vy, vz]
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels_no_bev, config=config) # return partial when initialization 

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE, # dataset voxel size. [0.1, 0.1, 0.15]
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points = data_dict['points']
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output # voxels: (n_voxel, max_n_point_per_voxel, C); coordinates: (n_voxel, 4[b_idx, z_idx, y_idx, x_idx])

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['voxels'] = voxels # should be features 
        data_dict['voxel_coords'] = coordinates # coordinates 
        data_dict['voxel_num_points'] = num_points # num points in side a voxel

        return data_dict
        
    def transform_points_to_voxels_dbscan(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels_dbscan, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points = data_dict['points']
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['voxels'] = voxels # should be (N, n_max_point_per_voxel, C)
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        # print(voxels.shape)

        ##############################

        self.voxel_generator_gt = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE_BEV,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL_BEV,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points_gt = data_dict['points']
        # print('before', points_gt.shape)
        # estimator = DBSCAN(eps=0.75,min_samples=5,metric='euclidean')
        # estimator.fit(points_gt[:, :3])

        # cls_idx = (estimator.labels_!=-1)
        plane1 = pyrsc.Plane()
        best_eq, best_inliers = plane1.fit(points_gt[:, :3], thresh=0.1)
        # print(cls_idx[:10])

        # points_gt = points_gt[cls_idx]
        idx = np.ones(points_gt.shape[0])
        idx[best_inliers] = 0
        idx = idx > 0
        points_gt = points_gt[idx]
        # points_gt = points_gt[best_inliers]
        # print('after', points_gt.shape)

        voxel_output_gt = self.voxel_generator_gt.generate(points_gt)
        voxels_gt, coordinates_gt, num_points_gt = voxel_output_gt

        # gt voxels for BEV reconstruction. 
        data_dict['voxels_bev'] = voxels_gt 
        data_dict['voxel_coords_bev'] = coordinates_gt
        data_dict['voxel_num_points_bev'] = num_points_gt
        # print(voxels_gt.shape, coordinates_gt.shape, num_points_gt.shape)
        # print(voxels_gt.shape)


        return data_dict

    def transform_points_to_voxels_PointToVoxel(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE) # (max range - min range) / voxel size per axis
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE # [vx, vy, vz]
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels_PointToVoxel, config=config) # return partial when initialization 

        points = data_dict['points']
        
        if self.voxel_generator is None:
            self.voxel_generator = PointToVoxel(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                # num_point_features=self.num_point_features,
                num_point_features=points.shape[1],
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL
            )

        
        points_th = torch.from_numpy(points)
        voxels_th, indices_th, num_p_in_vx_th, pc_voxel_id = self.voxel_generator.generate_voxel_with_id(
            points_th, empty_mean=False)
        # voxels: (n_voxel, max_n_point_per_voxel, C); 
        # coordinates: (n_voxel, 4[b_idx, z_idx, y_idx, x_idx]) 
        # num_points: (n_voxel, ) 
        # inverse mapping: (n_input_points, ). when point has no voxel assigned (e.g., not sampled), -1.  
        voxels = voxels_th.numpy()
        coordinates = indices_th.numpy()
        num_points = num_p_in_vx_th.numpy()
        inverse_mapping = pc_voxel_id.numpy() 

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['voxels'] = voxels # should be features 
        data_dict['voxel_coords'] = coordinates # coordinates 
        data_dict['voxel_num_points'] = num_points # num points in side a voxel
        data_dict['inverse_mapping'] = inverse_mapping
        # debug 
        if coordinates.shape[0] == 0:
            print(points.shape)
            np.save("zero_coord_points.npy", points)
        return data_dict


    def transform_points_to_voxels_PointToVoxel_with_BEV(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE) # (max range - min range) / voxel size per axis
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE # [vx, vy, vz]
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels_PointToVoxel_with_BEV, config=config) # return partial when initialization 

        
        points = data_dict['points']

        if self.voxel_generator is None:
            self.voxel_generator = PointToVoxel(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                # num_point_features=self.num_point_features,
                num_point_features=points.shape[1],
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL
            )

        
        points_th = torch.from_numpy(points)
        voxels_th, indices_th, num_p_in_vx_th, pc_voxel_id = self.voxel_generator.generate_voxel_with_id(
            points_th, empty_mean=False)
        # voxels: (n_voxel, max_n_point_per_voxel, C); 
        # coordinates: (n_voxel, 4[b_idx, z_idx, y_idx, x_idx]) 
        # num_points: (n_voxel, ) 
        # inverse mapping: (n_input_points, ). when point has no voxel assigned (e.g., not sampled), -1.  
        voxels = voxels_th.numpy()
        coordinates = indices_th.numpy()
        num_points = num_p_in_vx_th.numpy()
        inverse_mapping = pc_voxel_id.numpy() 

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['voxels'] = voxels # should be features 
        data_dict['voxel_coords'] = coordinates # coordinates 
        data_dict['voxel_num_points'] = num_points # num points in side a voxel
        data_dict['inverse_mapping'] = inverse_mapping

        # self.voxel_generator_gt = VoxelGeneratorWrapper(
        #         vsize_xyz=config.VOXEL_SIZE_BEV, # dataset voxel size for BEV: [0.8, 0.8, 6]
        #         coors_range_xyz=self.point_cloud_range,
        #         num_point_features=self.num_point_features,
        #         max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL_BEV,
        #         max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
        #     )

        points_gt = data_dict['points']
        self.voxel_generator_gt = PointToVoxel(
                vsize_xyz=config.VOXEL_SIZE_BEV,
                coors_range_xyz=self.point_cloud_range,
                # num_point_features=self.num_point_features,
                num_point_features=points_gt.shape[1],
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL_BEV
            )
        
        # points_gt = data_dict['points']
        # voxel_output_gt = self.voxel_generator_gt.generate(points_gt)
        # voxels_gt, coordinates_gt, num_points_gt = voxel_output_gt

        # data_dict['voxels_bev'] = voxels_gt
        # data_dict['voxel_coords_bev'] = coordinates_gt
        # data_dict['voxel_num_points_bev'] = num_points_gt
        
        points_gt_th = torch.from_numpy(points_gt)
        voxels_gt_th, indices_gt_th, num_p_in_vx_gt_th, pc_voxel_id_gt = self.voxel_generator_gt.generate_voxel_with_id(
            points_gt_th, empty_mean=False)
        # voxels: (n_voxel, max_n_point_per_voxel, C); 
        # coordinates: (n_voxel, 4[b_idx, z_idx, y_idx, x_idx]) 
        # num_points: (n_voxel, ) 
        # inverse mapping: (n_input_points, ). when point has no voxel assigned (e.g., not sampled), -1.  
        voxels_gt = voxels_gt_th.numpy()
        coordinates_gt = indices_gt_th.numpy()
        num_points_gt = num_p_in_vx_gt_th.numpy()
        inverse_mapping_gt = pc_voxel_id_gt.numpy() 

        data_dict['voxels_bev'] = voxels_gt
        data_dict['voxel_coords_bev'] = coordinates_gt
        data_dict['voxel_num_points_bev'] = num_points_gt
        data_dict['inverse_mapping_gt'] = inverse_mapping_gt

        return data_dict


    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def random_sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_sample_points, config=config)

        num_points = config.NUM_POINTS
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        len_points = len(points)
        if num_points < len_points:
            choice = np.random.choice(np.arange(len(points)), num_points, replace=False)
        # elif num_points > len_points:
        #     choice = np.random.choice(np.arange(len(points)), num_points, replace=True)
        else:
            choice = np.arange(len(points))
        data_dict['points'] = points[choice]
        data_dict['random_sample_indices'] = choice
        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)

        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict


