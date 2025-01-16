import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu

from pcdet.utils import common_utils

from scipy.spatial import KDTree
from sklearn.neighbors import KDTree

from scipy.sparse import csr_matrix
from scipy.stats import mode

import copy

from os.path import exists

import multiprocessing

def to_torch_tensor(t, device='cuda'):
    return torch.FloatTensor(t).to(device)

def interpolation(original_xyz, pred_xyz, pred, k=3):
    tree = KDTree(pred_xyz, leaf_size=2)
    dist, ind = tree.query(original_xyz, k=k)
    interpolated = pred[ind] # n, k, c
    w = 1 / (dist + 1e-6) # n, k
    w = w / np.sum(w)
    interpolated = np.sum(interpolated * w[..., None], 1) # n, c
    return interpolated

def average_factor_for_repeats(x):
    u,inv,c = np.unique(x, return_counts=True, return_inverse=True)
    div_factor = c[inv]
    return div_factor

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    # model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)

        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


def eval_one_epoch_seg_pointwise(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    # for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
    #     metric['recall_roi_%s' % str(cur_thresh)] = 0
    #     metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    num_classes = len(class_names)
    det_annos = []

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    
    cloud_predictions = []
    cloud_gts = []
    for i, batch_dict in enumerate(dataloader):
        # load_data_to_gpu(batch_dict)
        # data preparation: run for each element in the list to obtain a full prediction of a cloud. assuming batch size=1 
        voxel_list_full = batch_dict['voxels_list_full'] # list of: should be features (N, max_pts_per_voxel, num_feature)
        sampled_indices_list_full = batch_dict['sampled_indices_list_full'] # list of: (N, max_pts_per_voxel)
        voxel_coords_list_full = batch_dict['voxel_coords_list_full'] # list of: vox coordinates  (N, 3)
        voxel_num_points_list_full = batch_dict['voxel_num_points_list_full'] # list of: num unique points in side a voxel

        one_cloud_gt = batch_dict['point_seg_labels_full']
        cloud_gts.append(one_cloud_gt)
        
        pred_iter = len(voxel_list_full) # how many iterations needed to cover all points in the original cloud 
        pred_container = np.zeros(one_cloud_gt.shape[0], num_classes) # for aggregating the prediction scores
        for j in range(pred_iter):
            batch_dict['voxels'] = voxel_list_full[j] # should be features (N, max_pts_per_voxel, num_feature)
            batch_dict['sampled_indices'] = sampled_indices_list_full[j] # (N, max_pts_per_voxel)
            batch_dict['voxel_coords'] =  voxel_coords_list_full[j] # vox coordinates  (N, 3)
            batch_dict['voxel_num_points'] = voxel_num_points_list_full[j] # num unique points in side a voxel
            

            # import pdb; pdb.set_trace()

            if getattr(args, 'infer_time', False):
                start_time = time.time()

            with torch.no_grad(): #TODO: amp
                pred_dicts, _ = model(batch_dict)
            
            # aggregate predictions to the orignal point clouds 
            indice_part = batch_dict['sampled_indices'].detach().cpu().numpy() # n_part, max_pts_per_voxel
            num_repeat = indice_part.shape[1]
            pred_part = pred_dicts['pred'].detach().cpu().numpy() # n_part, num_classes
            pred_part_expand = np.repeat(pred_part[:, None, :], repeats=num_repeat, axis=1) # n_part, max_pts_per_voxel, num_classes
            div_factor = np.apply_along_axis(lambda x: average_factor_for_repeats(x), axis=1, arr=indice_part) # n_part, max_pts_per_voxel
            pred_part_expand /= div_factor
            pred_part_expand = pred_part_expand.reshape(-1, num_classes) # n_part * max_pts_per_voxel, num_classes
            unique_indices = np.unique(indice_part.reshape(-1))
            aggregated_preds = np.array([pred_part_expand[indice_part == idx].sum(0) for idx in unique_indices]) # should be (len(unique_dinces), num_classes)
            pred_container[unique_indices] += aggregated_preds

            disp_dict = {}

            if getattr(args, 'infer_time', False):
                inference_time = time.time() - start_time
                infer_time_meter.update(inference_time * 1000)
                # use ms to measure inference time
                disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

            # statistics_info(cfg, ret_dict, metric, disp_dict)
            # annos = dataset.generate_prediction_dicts(
            #     batch_dict, pred_dicts, class_names,
            #     output_path=final_output_dir if args.save_to_file else None
            # )
            # det_annos += annos
            if cfg.LOCAL_RANK == 0:
                progress_bar.set_postfix(disp_dict)
                progress_bar.update()
        
        # start evaluation
        one_cloud_prediction = np.argmax(pred_container, axis=1) # n, 1
        area_intersection, area_union, area_target = common_utils.intersectionAndUnion(one_cloud_prediction.reshape(-1), one_cloud_gt.reshape(-1))
        cloud_iou = area_intersection / area_union
        logger.info(f"{i}: mIoU: {np.mean(cloud_iou)}")
        for k, c_name in enumerate(class_names):
            logger.info(f"{c_name}: {cloud_iou[k]}")
        cloud_predictions.append(one_cloud_prediction)
        
        # save one cloud
        save_dict = {
            "points": batch_dict["points"],
            "pred": one_cloud_prediction,
            "gt": one_cloud_gt,
            "mIoU": np.mean(cloud_iou),
            "IoU": cloud_iou
        } 
        with open(result_dir / 'point_cloud_{i}.pkl', 'wb') as f:
            pickle.dump(save_dict, f)

    # evalute globally 
    global_prediction = np.concatenate(cloud_predictions, 0).reshape(-1)
    global_gt = np.concatenate(cloud_gts, 0).reshape(-1)
    area_intersection, area_union, area_target = common_utils.intersectionAndUnion(global_prediction, global_gt)
    IoUs = area_intersection / area_union
    

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    # if dist_test:
    #     rank, world_size = common_utils.get_dist_info()
    #     det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
    #     metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    logger.info(f"Global prediction: mIoU: {np.mean(IoUs)}")
    for k, c_name in enumerate(class_names):
        logger.info(f"{c_name}: {cloud_iou[k]}")


    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    # if dist_test:
    #     for key, val in metric[0].items():
    #         for k in range(1, world_size):
    #             metric[0][key] += metric[k][key]
    #     metric = metric[0]

    # gt_num_cnt = metric['gt_num']
    # for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
    #     cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
    #     cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
    #     logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
    #     logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
    #     ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
    #     ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    # total_pred_objects = 0
    # for anno in det_annos:
    #     total_pred_objects += anno['name'].__len__()
    # logger.info('Average predicted number of objects(%d samples): %.3f'
    #             % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    # with open(result_dir / 'result.pkl', 'wb') as f:
    #     pickle.dump(det_annos, f)

    # result_str, result_dict = dataset.evaluation(
    #     det_annos, class_names,
    #     eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
    #     output_path=final_output_dir
    # )

    # logger.info(result_str)
    # ret_dict.update(result_dict)

    # logger.info('Result is saved to %s' % result_dir)
    # logger.info('****************Evaluation done.*****************')
    return ret_dict

def eval_one_epoch_seg_interpolate(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    # for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
    #     metric['recall_roi_%s' % str(cur_thresh)] = 0
    #     metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    num_classes = len(class_names)
    det_annos = []

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    
    cloud_predictions = []
    cloud_gts = []
    for i, batch_dict in enumerate(dataloader):
        # load_data_to_gpu(batch_dict)
        # data preparation: run for each element in the list to obtain a full prediction of a cloud. assuming batch size=1 
        voxel_list_full = batch_dict['voxels_list_full'] # list of: should be features (N, max_pts_per_voxel, num_feature)
        sampled_indices_list_full = batch_dict['sampled_indices_list_full'] # list of: (N, max_pts_per_voxel)
        voxel_coords_list_full = batch_dict['voxel_coords_list_full'] # list of: vox coordinates  (N, 3)
        voxel_num_points_list_full = batch_dict['voxel_num_points_list_full'] # list of: num unique points in side a voxel

        one_cloud_gt = batch_dict['point_seg_labels_full']
        cloud_gts.append(one_cloud_gt)
        
        pred_iter = len(voxel_list_full) # how many iterations needed to cover all points in the original cloud 
        pred_container = np.zeros((one_cloud_gt.shape[0], num_classes)) # for aggregating the prediction scores
        predicted_indices = []

        b_idx = 0
        for j in range(pred_iter):
            batch_dict['voxels'] = to_torch_tensor(voxel_list_full[b_idx][j]) # should be features (N, max_pts_per_voxel, num_feature)
            batch_dict['sampled_indices'] = to_torch_tensor(sampled_indices_list_full[b_idx][j]) # (N, max_pts_per_voxel)
            batch_dict['voxel_coords'] =  to_torch_tensor(np.pad(voxel_coords_list_full[b_idx][j][:, ::-1], ((0, 0), (1, 0)), mode='constant', constant_values=0)) # vox coordinates  (N, 4)
            batch_dict['voxel_num_points'] = to_torch_tensor(voxel_num_points_list_full[b_idx][j]) # num unique points in side a voxel
            batch_dict['point_seg_labels'] = torch.FloatTensor([0]).cuda() # dummy sample labels 
            # import pdb; pdb.set_trace()

            if getattr(args, 'infer_time', False):
                start_time = time.time()

            with torch.no_grad(): #TODO: amp
                pred_dicts, _ = model(batch_dict)
            
            # aggregate predictions to the orignal point clouds 
            indice_part = batch_dict['sampled_indices'].detach().cpu().numpy() # n_part, max_pts_per_voxel
            num_repeat = indice_part.shape[1]
            pred_part = pred_dicts['pred'].detach().cpu().numpy() # n_part, num_classes
            pred_part_expand = np.repeat(pred_part[:, None, :], repeats=num_repeat, axis=1) # n_part, max_pts_per_voxel, num_classes
            div_factor = np.apply_along_axis(lambda x: average_factor_for_repeats(x), axis=1, arr=indice_part) # n_part, max_pts_per_voxel
            pred_part_expand = pred_part_expand / div_factor[:, :, None]
            pred_part_expand = pred_part_expand.reshape(-1, num_classes) # n_part * max_pts_per_voxel, num_classes
            indice_part_flatten = indice_part.reshape(-1).astype(int)
            unique_indices = np.unique(indice_part_flatten).astype(indice_part_flatten.dtype)
            
            # import pdb; pdb.set_trace()
            # aggregated_preds = np.array([pred_part_expand[indice_part_flatten == idx].sum(0) for idx in unique_indices]) # should be (len(unique_dinces), num_classes)
            # mask = (indice_part_flatten[:, None] == unique_indices).astype(np.float32)
            # aggregated_preds = mask.T @ pred_part_expand
            # Create a sparse matrix for the mask. this aggregate the prediction from all repeated indices by a linear transformation fashion. it is made sparse for memory efficiency. 
            rows = np.arange(indice_part_flatten.size)
            cols = np.searchsorted(unique_indices, indice_part_flatten)
            data = np.ones(indice_part_flatten.size)
            sparse_mask = csr_matrix((data, (rows, cols)), shape=(indice_part_flatten.size, unique_indices.size))

            # Use the sparse mask to aggregate the predictions
            aggregated_preds = sparse_mask.T @ pred_part_expand

            pred_container[unique_indices] += aggregated_preds
            predicted_indices.append(unique_indices)
            disp_dict = {}

            if getattr(args, 'infer_time', False):
                inference_time = time.time() - start_time
                infer_time_meter.update(inference_time * 1000)
                # use ms to measure inference time
                disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

            # statistics_info(cfg, ret_dict, metric, disp_dict)
            # annos = dataset.generate_prediction_dicts(
            #     batch_dict, pred_dicts, class_names,
            #     output_path=final_output_dir if args.save_to_file else None
            # )
            # det_annos += annos
            if cfg.LOCAL_RANK == 0:
                progress_bar.set_postfix(disp_dict)
                progress_bar.update()
        
        # start interpolation
        original_xyz = batch_dict["points"][:, :3] # n, 3
        predicted_indices = np.unique(np.concatenate(predicted_indices, 0))
        predicted_xyz = original_xyz[predicted_indices]
        one_cloud_prediction = interpolation(original_xyz, predicted_xyz, pred_container)
        assert one_cloud_prediction.shape[0] == original_xyz.shape[0]
        
        one_cloud_prediction = np.argmax(pred_container, axis=1) # n, 1
        area_intersection, area_union, area_target = common_utils.intersectionAndUnion(one_cloud_prediction.reshape(-1), one_cloud_gt.reshape(-1), num_classes)
        cloud_iou = area_intersection / area_union

        logger.info(f"{i}: mIoU: {np.mean(cloud_iou)}")
        for k, c_name in enumerate(class_names):
            logger.info(f"{c_name}: {cloud_iou[k]}")
        cloud_predictions.append(one_cloud_prediction)
        
        # save one cloud
        save_dict = {
            "points": batch_dict["points"],
            "pred": one_cloud_prediction,
            "gt": one_cloud_gt,
            "mIoU": np.mean(cloud_iou),
            "IoU": cloud_iou
        } 
        with open(result_dir / 'point_cloud_{i}.pkl', 'wb') as f:
            pickle.dump(save_dict, f)

    # evalute globally 
    global_prediction = np.concatenate(cloud_predictions, 0).reshape(-1)
    global_gt = np.concatenate(cloud_gts, 0).reshape(-1)
    area_intersection, area_union, area_target = common_utils.intersectionAndUnion(global_prediction, global_gt, num_classes)
    IoUs = area_intersection / area_union
    

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    # if dist_test:
    #     rank, world_size = common_utils.get_dist_info()
    #     det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
    #     metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    logger.info(f"Global prediction: mIoU: {np.mean(IoUs)}")
    for k, c_name in enumerate(class_names):
        logger.info(f"{c_name}: {cloud_iou[k]}")


    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    # if dist_test:
    #     for key, val in metric[0].items():
    #         for k in range(1, world_size):
    #             metric[0][key] += metric[k][key]
    #     metric = metric[0]

    # gt_num_cnt = metric['gt_num']
    # for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
    #     cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
    #     cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
    #     logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
    #     logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
    #     ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
    #     ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    # total_pred_objects = 0
    # for anno in det_annos:
    #     total_pred_objects += anno['name'].__len__()
    # logger.info('Average predicted number of objects(%d samples): %.3f'
    #             % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    # with open(result_dir / 'result.pkl', 'wb') as f:
    #     pickle.dump(det_annos, f)

    # result_str, result_dict = dataset.evaluation(
    #     det_annos, class_names,
    #     eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
    #     output_path=final_output_dir
    # )

    # logger.info(result_str)
    # ret_dict.update(result_dict)

    # logger.info('Result is saved to %s' % result_dir)
    # logger.info('****************Evaluation done.*****************')
    return ret_dict


def pillar_crop(data_dict=None, config=None, crop_center=None):
    # modifyied pillar crop so that the crop center can be passed. 
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

    # grid_points_2d = grid_sampling_2d(points[:, :2], grid_size)
    # crop_center = grid_points_2d[np.random.choice(np.arange(len(grid_points_2d)))]
    
    cropped, crop_indices = crop_point_cloud(points, crop_center, pillar_size)
    cropped[:, :2] = cropped[:, :2] - crop_center # centralize 
    data_dict['points'] = cropped
    data_dict['crop_indices'] = crop_indices
    return data_dict

def eval_one_epoch_seg_temp(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset
    class_names = dataset.class_names
    num_classes = len(class_names)

    model_inference_timer = common_utils.AverageMeter() # model inference speed

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )

    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    
    start_time = time.time()
    
    # load data 
    dataset = dataloader.dataset # get dataset class
    num_pc = dataloader.__len__() # all test samples (loop=1)
    logger.info(f"Total samples to be tested: {num_pc}")
    global_pred_list = [] # store predictions list(num_test_cloud)
    global_seg_list = [] # store ground truth labels list(num_test_cloud)
    
    
    # for each cloud
    for cloud_id, index in enumerate(range(num_pc)):
        # load each cloud
        info = copy.deepcopy(dataset.infos[index])
        pc_info = info['point_cloud']
        region_name = pc_info['region_name']
        sample_idx = pc_info['sample_idx']
        cloud_save_path = result_dir / f'{region_name}_{sample_idx}.pkl'
        if exists(cloud_save_path): # load the saved prediction
            logger.info(f"Loaded {cloud_save_path}")
            with open(cloud_save_path, 'rb') as f:
                save_dict = pickle.load(f) # load saved pickles
            # report the singel crop performance
            mIoU_single_cloud = save_dict["mIoU"]
            class_wise_iou_single_cloud = save_dict["class_wise_iou"]
            logger.info(f"\n==== Performance of {region_name}_{sample_idx} ====")
            logger.info(f"{region_name}_{sample_idx}: mIoU: {mIoU_single_cloud}")
            for k, c_name in enumerate(dataset.class_names):
                logger.info(f"{c_name}: {class_wise_iou_single_cloud[k]:.4f}")
            logger.info(f"\n====  ====")
            pred_single_cloud = save_dict["pred"]
            seg_single_cloud = save_dict["seg"]
            # update global pred and seg
            global_pred_list.append(pred_single_cloud)
            global_seg_list.append(seg_single_cloud)
            
            continue

        points = dataset.get_lidar(region_name, sample_idx)
        seg_single_cloud = points[:, -1]
        
        # generate crops from original big clouds. Each crop size = pillar size 
        # crop_size = dataset.data_processor.processor_configs.pillar_crop.PILLAR_SIZE
        crop_size= 256 # TODO: load from config pillar crop size
        stride = crop_size//4  # the amount of overlap
        crop_start_time = time.time()
        crop_list, crop_ind_list = generate_crops(points, crop_size, stride)
        crop_time = time.time() - crop_start_time
        flattened_crop_ind_list = [item for sublist in crop_ind_list for item in sublist]
        assert len(np.unique(flattened_crop_ind_list)) == len(points), f"the number of unique indices is {len(np.unique(flattened_crop_ind_list))}, while should have {len(points)}"  # make sure no points missing.
        logger.info(f"Number of crops for cloud {region_name}_{sample_idx}: {len(crop_ind_list)}. Crop size {crop_size}, stride: {stride}, took {crop_time:.2f} seconds for cropping.")

        # for each crop 
        logit_single_cloud = torch.zeros((points.shape[0], len(dataset.class_names))).cuda() # container for predictions on this single crop
        for crop_id, (single_crop, single_crop_ind) in enumerate(zip(crop_list, crop_ind_list)):
            # generate idx_data that covers all 
            coord = single_crop[:, :3]
            feat = single_crop[:, 3:-1]
            label = single_crop[:, -1]
            idx_data = []
            coord_min = np.min(coord, 0)
            coord = coord - coord_min
            
            # the voxel size for each sample. 
            # Should be smaller than traiing voxel size so that denser points can be obtained 
            # voxel_size_train = dataset.data_processor.processor_configs.transform_points_to_voxels_numpy.VOXEL_SIZE
            # voxel_size = voxel_size_train * 0.2 
            voxel_size = 0.2 # TODO load from config
            voxelize_start_time = time.time()
            idx_sort, count = voxelize(coord, voxel_size, mode=1) # validation mode 
            voxelize_time = time.time() - voxelize_start_time
            for i in range(count.max()): # ensure all points within a voxle are taken at least once
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                idx_data.append(idx_part)
            
            logger.info(f"There are {len(idx_data)} partial crops for crop {crop_id+1}/{len(crop_list)}. Voxelizations took {voxelize_time:.2f} sconds")
            
            # each element in idx_data is a cropped tile voxelized by voxel_size. 
            # pointcept like sphere sampling is not implemented furhter. 
            # will use inverse mapping to obtain the full predictions for crops

            # for each partial crop, prepare data and create batches 
            logit_single_crop = torch.zeros((label.size, len(dataset.class_names))).cuda() # container for predictions on this single crop
            
            # accumulate prepareed data of a partial crop 
            logger.info(f"Start preparing data for crop {crop_id+1}/{len(crop_list)}")
            data_dict_list = []
            data_prep_start_time  = time.time()
            for partial_crop_id, partial_index in enumerate(idx_data):
                
                # logger.info(f"partial crop, crop, cloud: {partial_crop_id+1}/{len(idx_data)}, {crop_id+1}/{len(crop_list)}, {cloud_id+1}/{num_pc}")
                
                partial_crop = single_crop[partial_index]
                partial_points = partial_crop[:, :-1]
                partial_seg = partial_crop[:, -1]
                input_dict = {
                    "points": partial_points,
                    "point_seg_labels_full": partial_seg 
                }
                
                # prepare data (emulate __getitem()__ in dataset)

                # point feature encoder 
                data_dict = dataset.point_feature_encoder.forward(input_dict)

                # data processor
                data_processor_queue = dataset.data_processor.data_processor_queue
                for cur_processor in data_processor_queue:
                    if cur_processor.func.__name__ == 'pillar_crop': # already cropped. 
                        # print(f"{cur_processor.func.__name__} skipped.")
                        continue
                    data_dict = cur_processor(data_dict=data_dict)
                
                # get shuffle and inverse mapping such that the original order can be recovered. 
                shuffle_idx = data_dict.pop('shuffle_idx', None) # because data processing invovles shuffling
                shuffle_idx_inverse = np.argsort(shuffle_idx) # to recover the original order later. 
                data_dict['shuffle_idx_inverse'] = shuffle_idx_inverse
                # inverse_mapping = data_dict.pop('inverse_mapping', None) # this will map the crop prediction back to shuffled orignal partial crop
                
                random_sample_idx = data_dict.pop('random_sample_indices', None)
                sampled_partial_seg = partial_seg[shuffle_idx] if shuffle_idx is not None else partial_seg
                sampled_partial_seg = sampled_partial_seg[random_sample_idx] if random_sample_idx is not None else sampled_partial_seg
                sampled_indices = data_dict['sampled_indices']
                sampled_partial_seg_grouped = sampled_partial_seg[sampled_indices]
                sampled_partial_seg_aggregated = mode(sampled_partial_seg_grouped, axis=1, keepdims=False).mode
                data_dict['point_seg_labels'] = sampled_partial_seg_aggregated.reshape(-1, 1)
                data_dict_list.append(data_dict)
            data_prep_time = time.time() - data_prep_start_time

            logger.info(f"Start prediction for the crop {crop_id+1}/{len(crop_list)}. Preparation took {data_prep_time:.2f} seconds.")
            batch_size_test = 16 # TODO set by hyperparam
            batch_num = int(np.ceil(len(data_dict_list) / batch_size_test))
            for batch_i in range(batch_num):
                # prepare input and model prediction 
                s_i, e_i = batch_i * batch_size_test, min((batch_i + 1) * batch_size_test, len(data_dict_list))
                data_dict = data_dict_list[s_i:e_i]
                partial_indices = idx_data[s_i:e_i]
                # collate
                if not isinstance(data_dict, list):
                    data_dict = [data_dict]
                    partial_indices = [partial_indices]
                batch_dict = dataset.collate_batch(data_dict)
                internal_batch_size = batch_dict["batch_size"]
                # convert the batch into torch tensor
                load_data_to_gpu(batch_dict)# in "model_func_decorator"

                # with torch.cuda.amp.autocast(enabled=use_amp):
                model.eval()
                model_forward_start = time.time()
                with torch.no_grad():
                   ret_dict, tb_dict, disp_dict = model(batch_dict)
                model_forward_time = time.time() - model_forward_start
                model_inference_timer.update(model_forward_time/internal_batch_size)
                
                # just for reference to see if scores are normal
                batch_mIoU = ret_dict["batch_mIoU"]
                loss = ret_dict["loss"]
                
                # handle predictions
                internal_batch_size = batch_dict["batch_size"]
                internal_batch_idx_sampled =  batch_dict["voxel_coords"][:, 0] # prediction
                internal_batch_idx_full =  batch_dict["points"][:, 0] # inverse mapping, shuffle_inverse_idx
                for b_i in range(internal_batch_size):
                    # recover partial crop in its original order by interating the batch
                    # import pdb; pdb.set_trace()
                    selected_sampled = internal_batch_idx_sampled == b_i
                    selected_full = internal_batch_idx_full == b_i
                    sampled_partial_logits = disp_dict["pred"][selected_sampled] # logits
                    inverse_mapping = batch_dict["inverse_mapping"][selected_full] # (batch_size * n_point_of_sample, )
                    # print(inverse_mapping)
                    shuffle_idx_inverse = batch_dict["shuffle_idx_inverse"][selected_full]
                    partial_logits_shuffled = sampled_partial_logits[inverse_mapping] # upsample by inverse mapping
                    partial_logits = partial_logits_shuffled[shuffle_idx_inverse] # unshuffle

                    # add to the prediction 
                    partial_index = partial_indices[b_i]
                    logit_single_crop[partial_index, :] += partial_logits
                # logger.info(f"  batch mIoU: {batch_mIoU:.4f}, loss: {loss:.4f}, model speed: {model_inference_timer.avg:.2f} sec/batch")
                logger.info(f"batch mIoU: {batch_mIoU:.4f}, loss: {loss:.4f}, model speed: {model_inference_timer.avg:.2f} sec/batch\n \
                            partial crop, crop, cloud: {e_i}/{len(idx_data)}, {crop_id+1}/{len(crop_list)}, {cloud_id+1}/{num_pc}")
            # here finished for a single crop 
            # now assign predictions to the original cloud by crop --> original cloud mapping.
            logit_single_cloud[single_crop_ind, :] += logit_single_crop
        
        # here finished predictions for a single cloud 
        logit_single_cloud = logit_single_cloud.detach().cpu().numpy()
        pred_single_cloud = np.argmax(logit_single_cloud, axis=1) # prediction: highest score

        # single cloud evaluation 
        area_intersection, area_union, area_target = common_utils.intersectionAndUnion(
            pred_single_cloud.reshape(-1), seg_single_cloud.reshape(-1), 
            len(dataset.class_names), ignore_index=dataset.ignore_index
        )
        class_wise_iou_single_cloud = area_intersection / (area_union + 1e-8)
        mIoU_single_cloud = np.mean(class_wise_iou_single_cloud)
        
        # report metrics for a single cloud
        logger.info(f"\n==== Performance of {region_name}_{sample_idx} ====")
        logger.info(f"{region_name}_{sample_idx}: mIoU: {mIoU_single_cloud}")
        for k, c_name in enumerate(dataset.class_names):
            logger.info(f"{c_name}: {class_wise_iou_single_cloud[k]:.4f}")
        logger.info(f"\n====  ====")
        # save this single cloud to the disk  
        save_dict = {
            "points": batch_dict["points"].detach().cpu().numpy(),
            "pred": pred_single_cloud,
            "seg": seg_single_cloud,
            "mIoU": mIoU_single_cloud,
            "class_wise_iou": class_wise_iou_single_cloud
        } 
        logger.info(f"Saving {region_name}_{sample_idx}")
        with open(result_dir / f'{region_name}_{sample_idx}.pkl', 'wb') as f:
            pickle.dump(save_dict, f)
        
        # update the global pred and gt lists
        global_pred_list.append(pred_single_cloud)
        global_seg_list.append(seg_single_cloud)
    

    # here finished all clouds
    # start global evaluation 

    global_pred = np.concatenate(global_pred_list)
    global_seg = np.concatenate(global_seg_list)
    area_intersection, area_union, area_target = common_utils.intersectionAndUnion(
        global_pred.reshape(-1), global_seg.reshape(-1), 
        len(dataset.class_names), ignore_index=dataset.ignore_index
    )
    class_wise_iou_global = area_intersection / (area_union + 1e-8)
    mIoU_global = np.mean(class_wise_iou_global)
    
    # report metrics for a single cloud
    logger.info(f"\n======= Global performance: mIoU: {mIoU_global}========")
    for k, c_name in enumerate(dataset.class_names):
        logger.info(f"{c_name}: {class_wise_iou_global[k]:.4f}")

    logger.info("Saving global predictions and gts")
    save_dict = {
            "pred": global_pred,
            "seg": global_seg,
            "mIoU": mIoU_global,
            "class_wise_iou": class_wise_iou_global
        } 
    with open(result_dir / 'all_results.pkl', 'wb') as f:
        pickle.dump(save_dict, f)

    test_time = time.time() - start_time
    logger.info(f"Testing took {test_time:.2f} second, average model inference time: {model_inference_timer.avg:.2f} second/batch")

    ret_dict = save_dict
    return ret_dict

def eval_one_epoch_seg_parallel_prep(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset
    class_names = dataset.class_names
    num_classes = len(class_names)

    model_inference_timer = common_utils.AverageMeter() # model inference speed

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )

    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    
    start_time = time.time()
    
    # load data 
    dataset = dataloader.dataset # get dataset class
    num_pc = dataloader.__len__() # all test samples (loop=1)
    logger.info(f"Total samples to be tested: {num_pc}")
    global_pred_list = [] # store predictions list(num_test_cloud)
    global_seg_list = [] # store ground truth labels list(num_test_cloud)
    
    
    # for each cloud
    for cloud_id, index in enumerate(range(num_pc)):
        # load each cloud
        info = copy.deepcopy(dataset.infos[index])
        pc_info = info['point_cloud']
        region_name = pc_info['region_name']
        sample_idx = pc_info['sample_idx']
        cloud_save_path = result_dir / f'{region_name}_{sample_idx}.pkl'
        if exists(cloud_save_path): # load the saved prediction
            logger.info(f"Loaded {cloud_save_path}")
            with open(cloud_save_path, 'rb') as f:
                save_dict = pickle.load(f) # load saved pickles
            # report the singel crop performance
            mIoU_single_cloud = save_dict["mIoU"]
            class_wise_iou_single_cloud = save_dict["class_wise_iou"]
            logger.info(f"\n==== Performance of {region_name}_{sample_idx} ====")
            logger.info(f"{region_name}_{sample_idx}: mIoU: {mIoU_single_cloud}")
            for k, c_name in enumerate(dataset.class_names):
                logger.info(f"{c_name}: {class_wise_iou_single_cloud[k]:.4f}")
            logger.info(f"\n====  ====")
            pred_single_cloud = save_dict["pred"]
            seg_single_cloud = save_dict["seg"]
            # update global pred and seg
            global_pred_list.append(pred_single_cloud)
            global_seg_list.append(seg_single_cloud)
            
            continue

        points = dataset.get_lidar(region_name, sample_idx)
        seg_single_cloud = points[:, -1]
        
        # generate crops from original big clouds. Each crop size = pillar size 
        # crop_size = dataset.data_processor.processor_configs.pillar_crop.PILLAR_SIZE
        crop_size= 256 # TODO: load from config pillar crop size
        stride = crop_size//4  # the amount of overlap
        crop_start_time = time.time()
        crop_list, crop_ind_list = generate_crops(points, crop_size, stride)
        crop_time = time.time() - crop_start_time
        flattened_crop_ind_list = [item for sublist in crop_ind_list for item in sublist]
        assert len(np.unique(flattened_crop_ind_list)) == len(points), f"the number of unique indices is {len(np.unique(flattened_crop_ind_list))}, while should have {len(points)}"  # make sure no points missing.
        logger.info(f"Number of crops for cloud {region_name}_{sample_idx}: {len(crop_ind_list)}. Crop size {crop_size}, stride: {stride}, took {crop_time:.2f} seconds for cropping.")

        # for each crop 
        logit_single_cloud = torch.zeros((points.shape[0], len(dataset.class_names))).cuda() # container for predictions on this single crop
        for crop_id, (single_crop, single_crop_ind) in enumerate(zip(crop_list, crop_ind_list)):
            # generate idx_data that covers all 
            coord = single_crop[:, :3]
            feat = single_crop[:, 3:-1]
            label = single_crop[:, -1]
            idx_data = []
            coord_min = np.min(coord, 0)
            coord = coord - coord_min
            
            # the voxel size for each sample. 
            # Should be smaller than traiing voxel size so that denser points can be obtained 
            # voxel_size_train = dataset.data_processor.processor_configs.transform_points_to_voxels_numpy.VOXEL_SIZE
            # voxel_size = voxel_size_train * 0.2 
            voxel_size = 0.2 # TODO load from config
            voxelize_start_time = time.time()
            idx_sort, count = voxelize(coord, voxel_size, mode=1) # validation mode 
            voxelize_time = time.time() - voxelize_start_time
            for i in range(count.max()): # ensure all points within a voxle are taken at least once
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                idx_data.append(idx_part)
            
            logger.info(f"There are {len(idx_data)} partial crops for crop {crop_id+1}/{len(crop_list)}. Voxelizations took {voxelize_time:.2f} sconds")
            
            # each element in idx_data is a cropped tile voxelized by voxel_size. 
            # pointcept like sphere sampling is not implemented furhter. 
            # will use inverse mapping to obtain the full predictions for crops

            # for each partial crop, prepare data and create batches 
            logit_single_crop = torch.zeros((label.size, len(dataset.class_names))).cuda() # container for predictions on this single crop
            
            # accumulate prepareed data of a partial crop 
            logger.info(f"Start preparing data for crop {crop_id+1}/{len(crop_list)}")
            data_dict_list = []
            data_prep_start_time  = time.time()

            

            def parallel_processing_multiprocessing(idx_data, single_crop, dataset, num_workers=None):
                data_dict_list = []
                
                # Prepare the argument list for each process
                args_list = [(partial_crop_id, partial_index, idx_data, single_crop, dataset) 
                            for partial_crop_id, partial_index in enumerate(idx_data)]
                
                # Use multiprocessing Pool
                with multiprocessing.Pool(processes=num_workers, maxtasksperchild=1) as pool:
                    # Distribute the tasks across workers
                    results = pool.map(process_partial_crop, args_list)
                
                # Collect results
                data_dict_list.extend(results)
                
                return data_dict_list

            data_dict_list = parallel_processing_multiprocessing(idx_data, single_crop, dataset, num_workers=16)

            data_prep_time = time.time() - data_prep_start_time

            logger.info(f"Start prediction for the crop {crop_id+1}/{len(crop_list)}. Preparation took {data_prep_time:.2f} seconds.")
            batch_size_test = 16 # TODO set by hyperparam
            batch_num = int(np.ceil(len(data_dict_list) / batch_size_test))
            for batch_i in range(batch_num):
                # prepare input and model prediction 
                s_i, e_i = batch_i * batch_size_test, min((batch_i + 1) * batch_size_test, len(data_dict_list))
                data_dict = data_dict_list[s_i:e_i]
                partial_indices = idx_data[s_i:e_i]
                # collate
                if not isinstance(data_dict, list):
                    data_dict = [data_dict]
                    partial_indices = [partial_indices]
                batch_dict = dataset.collate_batch(data_dict)
                internal_batch_size = batch_dict["batch_size"]
                # convert the batch into torch tensor
                load_data_to_gpu(batch_dict)# in "model_func_decorator"

                # with torch.cuda.amp.autocast(enabled=use_amp):
                model.eval()
                model_forward_start = time.time()
                with torch.no_grad():
                   ret_dict, tb_dict, disp_dict = model(batch_dict)
                model_forward_time = time.time() - model_forward_start
                model_inference_timer.update(model_forward_time/internal_batch_size)
                
                # just for reference to see if scores are normal
                batch_mIoU = ret_dict["batch_mIoU"]
                loss = ret_dict["loss"]
                
                # handle predictions
                internal_batch_size = batch_dict["batch_size"]
                internal_batch_idx_sampled =  batch_dict["voxel_coords"][:, 0] # prediction
                internal_batch_idx_full =  batch_dict["points"][:, 0] # inverse mapping, shuffle_inverse_idx
                for b_i in range(internal_batch_size):
                    # recover partial crop in its original order by interating the batch
                    # import pdb; pdb.set_trace()
                    selected_sampled = internal_batch_idx_sampled == b_i
                    selected_full = internal_batch_idx_full == b_i
                    sampled_partial_logits = disp_dict["pred"][selected_sampled] # logits
                    inverse_mapping = batch_dict["inverse_mapping"][selected_full] # (batch_size * n_point_of_sample, )
                    # print(inverse_mapping)
                    shuffle_idx_inverse = batch_dict["shuffle_idx_inverse"][selected_full]
                    partial_logits_shuffled = sampled_partial_logits[inverse_mapping] # upsample by inverse mapping
                    partial_logits = partial_logits_shuffled[shuffle_idx_inverse] # unshuffle

                    # add to the prediction 
                    partial_index = partial_indices[b_i]
                    logit_single_crop[partial_index, :] += partial_logits
                # logger.info(f"  batch mIoU: {batch_mIoU:.4f}, loss: {loss:.4f}, model speed: {model_inference_timer.avg:.2f} sec/batch")
                logger.info(f"batch mIoU: {batch_mIoU:.4f}, loss: {loss:.4f}, model speed: {model_inference_timer.avg:.2f} sec/batch\n \
                            partial crop, crop, cloud: {e_i}/{len(idx_data)}, {crop_id+1}/{len(crop_list)}, {cloud_id+1}/{num_pc}")
            # here finished for a single crop 
            # now assign predictions to the original cloud by crop --> original cloud mapping.
            logit_single_cloud[single_crop_ind, :] += logit_single_crop
        
        # here finished predictions for a single cloud 
        logit_single_cloud = logit_single_cloud.detach().cpu().numpy()
        pred_single_cloud = np.argmax(logit_single_cloud, axis=1) # prediction: highest score

        # single cloud evaluation 
        area_intersection, area_union, area_target = common_utils.intersectionAndUnion(
            pred_single_cloud.reshape(-1), seg_single_cloud.reshape(-1), 
            len(dataset.class_names), ignore_index=dataset.ignore_index
        )
        class_wise_iou_single_cloud = area_intersection / (area_union + 1e-8)
        mIoU_single_cloud = np.mean(class_wise_iou_single_cloud)
        
        # report metrics for a single cloud
        logger.info(f"\n==== Performance of {region_name}_{sample_idx} ====")
        logger.info(f"{region_name}_{sample_idx}: mIoU: {mIoU_single_cloud}")
        for k, c_name in enumerate(dataset.class_names):
            logger.info(f"{c_name}: {class_wise_iou_single_cloud[k]:.4f}")
        logger.info(f"\n====  ====")
        # save this single cloud to the disk  
        save_dict = {
            "points": batch_dict["points"].detach().cpu().numpy(),
            "pred": pred_single_cloud,
            "seg": seg_single_cloud,
            "mIoU": mIoU_single_cloud,
            "class_wise_iou": class_wise_iou_single_cloud
        } 
        logger.info(f"Saving {region_name}_{sample_idx}")
        with open(result_dir / f'{region_name}_{sample_idx}.pkl', 'wb') as f:
            pickle.dump(save_dict, f)
        
        # update the global pred and gt lists
        global_pred_list.append(pred_single_cloud)
        global_seg_list.append(seg_single_cloud)
    

    # here finished all clouds
    # start global evaluation 

    global_pred = np.concatenate(global_pred_list)
    global_seg = np.concatenate(global_seg_list)
    area_intersection, area_union, area_target = common_utils.intersectionAndUnion(
        global_pred.reshape(-1), global_seg.reshape(-1), 
        len(dataset.class_names), ignore_index=dataset.ignore_index
    )
    class_wise_iou_global = area_intersection / (area_union + 1e-8)
    mIoU_global = np.mean(class_wise_iou_global)
    
    # report metrics for a single cloud
    logger.info(f"\n======= Global performance: mIoU: {mIoU_global}========")
    for k, c_name in enumerate(dataset.class_names):
        logger.info(f"{c_name}: {class_wise_iou_global[k]:.4f}")

    logger.info("Saving global predictions and gts")
    save_dict = {
            "pred": global_pred,
            "seg": global_seg,
            "mIoU": mIoU_global,
            "class_wise_iou": class_wise_iou_global
        } 
    with open(result_dir / 'all_results.pkl', 'wb') as f:
        pickle.dump(save_dict, f)

    test_time = time.time() - start_time
    logger.info(f"Testing took {test_time:.2f} second, average model inference time: {model_inference_timer.avg:.2f} second/batch")

    ret_dict = save_dict
    return ret_dict

def eval_one_epoch_seg_temp_v2(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    # this is the version which the model perform inverse mapping by itself
    # PointToVoxel is adopted for transforming points to voxels.
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset
    class_names = dataset.class_names
    num_classes = len(class_names)

    model_inference_timer = common_utils.AverageMeter() # model inference speed

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )

    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    
    start_time = time.time()
    
    # load data 
    dataset = dataloader.dataset # get dataset class
    num_pc = dataloader.__len__() # all test samples (loop=1)
    logger.info(f"Total samples to be tested: {num_pc}")
    global_pred_list = [] # store predictions list(num_test_cloud)
    global_seg_list = [] # store ground truth labels list(num_test_cloud)
    
    
    # for each cloud
    per_cloud_msg = []
    for cloud_id, index in enumerate(range(num_pc)):
        # load each cloud
        info = copy.deepcopy(dataset.infos[index])
        pc_info = info['point_cloud']
        region_name = pc_info['region_name']
        sample_idx = pc_info['sample_idx']
        cloud_save_path = result_dir / f'{region_name}_{sample_idx}.pkl'
        if exists(cloud_save_path): # load the saved prediction
            logger.info(f"Loaded {cloud_save_path}")
            with open(cloud_save_path, 'rb') as f:
                save_dict = pickle.load(f) # load saved pickles
            # report the singel crop performance
            mIoU_single_cloud = save_dict["mIoU"]
            class_wise_iou_single_cloud = save_dict["class_wise_iou"]
            logger.info(f"\n==== Performance of {region_name}_{sample_idx} ====")
            logger.info(f"{region_name}_{sample_idx}: mIoU: {mIoU_single_cloud}")
            per_cloud_msg.append(f"{region_name}_{sample_idx}: mIoU: {mIoU_single_cloud}")
            for k, c_name in enumerate(dataset.class_names):
                logger.info(f"{c_name}: {class_wise_iou_single_cloud[k]:.4f}")
            logger.info(f"\n====  ====")
            pred_single_cloud = save_dict["pred"]
            seg_single_cloud = save_dict["seg"]
            # update global pred and seg
            global_pred_list.append(pred_single_cloud)
            global_seg_list.append(seg_single_cloud)
            
            continue

        points = dataset.get_lidar(region_name, sample_idx)
        seg_single_cloud = points[:, -1]
        
        # generate crops from original big clouds. Each crop size = pillar size 
        # crop_size = dataset.data_processor.processor_configs.pillar_crop.PILLAR_SIZE
        data_processor_list = cfg.DATA_CONFIG.DATA_PROCESSOR
        found_pillar_crop = False
        for dict_element in data_processor_list:
            if dict_element["NAME"] == 'pillar_crop':
                found_pillar_crop = True
                crop_size = dict_element["PILLAR_SIZE"]
                break
        if not found_pillar_crop:
            raise ValueError("Could not find pillar crop in data processor")
        # crop_size = int(cfg.DATA_CONFIG.DATA_PROCESSOR[1].PILLAR_SIZE)
        if points.shape[0] > 40000000: 
            # for excessively large cloud, make slide smaller 
            stride = crop_size//2 # the amount of overlap
        else:
            stride = crop_size//4  
        crop_start_time = time.time()
        logger.info(f"Generating crops for {region_name} {sample_idx}")
        crop_list, crop_ind_list = generate_crops(points, crop_size, stride)
        crop_time = time.time() - crop_start_time
        flattened_crop_ind_list = [item for sublist in crop_ind_list for item in sublist]
        assert len(np.unique(flattened_crop_ind_list)) == len(points), f"the number of unique indices is {len(np.unique(flattened_crop_ind_list))}, while should have {len(points)}"  # make sure no points missing.
        logger.info(f"Number of crops for cloud {region_name}_{sample_idx}: {len(crop_ind_list)}. Crop size {crop_size}, stride: {stride}, took {crop_time:.2f} seconds for cropping.")

        # for each crop 
        logit_single_cloud = torch.zeros((points.shape[0], len(dataset.class_names))).cuda() # container for predictions on this single crop
        for crop_id, (single_crop, single_crop_ind) in enumerate(zip(crop_list, crop_ind_list)):
            # generate idx_data that covers all 
            coord = single_crop[:, :3]
            feat = single_crop[:, 3:-1]
            label = single_crop[:, -1]
            idx_data = []
            coord_min = np.min(coord, 0)
            coord = coord - coord_min
            
            # the voxel size for each sample. 
            # Should be smaller than traiing voxel size so that denser points can be obtained 
            # voxel_size_train = dataset.data_processor.processor_configs.transform_points_to_voxels_numpy.VOXEL_SIZE
            # voxel_size = voxel_size_train * 0.2 
            # voxel_size = 0.2 
            voxel_size = cfg.DATA_CONFIG.VOXEL_SIZE # usually 0.6
            voxel_size = voxel_size / 3 # 0.6 / 3 = 0.2
            logger.info(f"Voxel size is {voxel_size}")
            voxelize_start_time = time.time()
            idx_sort, count = voxelize(coord, voxel_size, mode=1) # validation mode 
            voxelize_time = time.time() - voxelize_start_time
            for i in range(count.max()): # ensure all points within a voxle are taken at least once
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                idx_data.append(idx_part)
            
            logger.info(f"There are {len(idx_data)} partial crops for crop {crop_id+1}/{len(crop_list)}. Voxel size: {voxel_size}. Voxelizations took {voxelize_time:.2f} sconds")
            
            # each element in idx_data is a cropped tile voxelized by voxel_size. 
            # pointcept like sphere sampling is not implemented furhter. 
            # will use inverse mapping to obtain the full predictions for crops

            # for each partial crop, prepare data and create batches 
            logit_single_crop = torch.zeros((label.size, len(dataset.class_names))).cuda() # container for predictions on this single crop
            
            # accumulate prepareed data of a partial crop 
            logger.info(f"Start preparing data for crop {crop_id+1}/{len(crop_list)}")
            data_dict_list = []
            data_prep_start_time  = time.time()
            for partial_crop_id, partial_index in enumerate(idx_data):
                
                # logger.info(f"partial crop, crop, cloud: {partial_crop_id+1}/{len(idx_data)}, {crop_id+1}/{len(crop_list)}, {cloud_id+1}/{num_pc}")
                
                partial_crop = single_crop[partial_index]
                partial_points = partial_crop[:, :-1]
                partial_seg = partial_crop[:, -1]
                input_dict = {
                    "points": partial_points,
                    "point_seg_labels_full": partial_seg 
                }
                
                # prepare data (emulate __getitem()__ in dataset)

                # point feature encoder 
                data_dict = dataset.point_feature_encoder.forward(input_dict)

                # data processor
                data_processor_queue = dataset.data_processor.data_processor_queue
                for cur_processor in data_processor_queue:
                    if cur_processor.func.__name__ == 'pillar_crop': # already cropped. 
                        # print(f"{cur_processor.func.__name__} skipped.")
                        continue
                    data_dict = cur_processor(data_dict=data_dict)
                
                # get shuffle and inverse mapping such that the original order can be recovered. 
                shuffle_idx = data_dict.pop('shuffle_idx', None) # because data processing invovles shuffling
                shuffle_idx_inverse = np.argsort(shuffle_idx) # to recover the original order later. 
                data_dict['shuffle_idx_inverse'] = shuffle_idx_inverse
                # inverse_mapping = data_dict.pop('inverse_mapping', None) # this will map the crop prediction back to shuffled orignal partial crop
                
                random_sample_idx = data_dict.pop('random_sample_indices', None)
                sampled_partial_seg = partial_seg[shuffle_idx] if shuffle_idx is not None else partial_seg
                sampled_partial_seg = sampled_partial_seg[random_sample_idx] if random_sample_idx is not None else sampled_partial_seg
                inverse_mapping = data_dict.get('inverse_mapping', None)
                if inverse_mapping is not None:
                    valid_mask = inverse_mapping != -1 # this shoud not contain -1
                    # import pdb; pdb.set_trace()
                    assert np.all(valid_mask), "mask contains invalid points. This probably means that the voxels in PointToVoxel fail to cover the entire input."
                    sampled_partial_seg = sampled_partial_seg[valid_mask]
                # sampled_indices = data_dict['sampled_indices']
                # sampled_partial_seg_grouped = sampled_partial_seg[sampled_indices]
                # sampled_partial_seg_aggregated = mode(sampled_partial_seg_grouped, axis=1, keepdims=False).mode
                # data_dict['point_seg_labels'] = sampled_partial_seg_aggregated.reshape(-1, 1)
                data_dict['point_seg_labels'] = sampled_partial_seg.reshape(-1, 1)
                data_dict_list.append(data_dict)
            data_prep_time = time.time() - data_prep_start_time

            logger.info(f"Start prediction for the crop {crop_id+1}/{len(crop_list)}. Preparation took {data_prep_time:.2f} seconds.")
            batch_size_test = 1 # TODO set by hyperparam
            batch_num = int(np.ceil(len(data_dict_list) / batch_size_test))
            for batch_i in range(batch_num):
                # prepare input and model prediction 
                s_i, e_i = batch_i * batch_size_test, min((batch_i + 1) * batch_size_test, len(data_dict_list))
                data_dict = data_dict_list[s_i:e_i]
                partial_indices = idx_data[s_i:e_i]
                # collate
                if not isinstance(data_dict, list):
                    data_dict = [data_dict]
                    partial_indices = [partial_indices]
                batch_dict = dataset.collate_batch(data_dict)
                internal_batch_size = batch_dict["batch_size"]
                # convert the batch into torch tensor
                load_data_to_gpu(batch_dict)# in "model_func_decorator"

                # with torch.cuda.amp.autocast(enabled=use_amp):
                model.eval()
                model_forward_start = time.time()
                with torch.no_grad():
                   ret_dict, tb_dict, disp_dict = model(batch_dict)
                model_forward_time = time.time() - model_forward_start
                model_inference_timer.update(model_forward_time/internal_batch_size)
                
                # just for reference to see if scores are normal
                batch_mIoU = ret_dict["batch_mIoU"]
                loss = ret_dict["loss"]
                
                # handle predictions
                internal_batch_size = batch_dict["batch_size"]
                internal_batch_idx_sampled = batch_dict["voxel_coords"][:, 0] # prediction
                internal_batch_idx_full =  batch_dict["points"][:, 0] # inverse mapping, shuffle_inverse_idx
                for b_i in range(internal_batch_size):
                    # recover partial crop in its original order by interating the batch
                    # import pdb; pdb.set_trace()
                    selected_full = internal_batch_idx_full == b_i
                    
                    sampled_partial_logits = disp_dict["pred"][selected_full] # logits
                    shuffle_idx_inverse = batch_dict["shuffle_idx_inverse"][selected_full]
                    partial_logits = sampled_partial_logits[shuffle_idx_inverse] # unshuffle

                    # add to the prediction 
                    partial_index = partial_indices[b_i]
                    logit_single_crop[partial_index, :] += partial_logits
                # logger.info(f"  batch mIoU: {batch_mIoU:.4f}, loss: {loss:.4f}, model speed: {model_inference_timer.avg:.2f} sec/batch")
                logger.info(f"batch mIoU: {batch_mIoU:.4f}, loss: {loss:.4f}, model speed: {model_inference_timer.avg:.2f} sec/batch\n \
                            partial crop, crop, cloud: {e_i}/{len(idx_data)}, {crop_id+1}/{len(crop_list)}, {cloud_id+1}/{num_pc}")
            # here finished for a single crop 
            # now assign predictions to the original cloud by crop --> original cloud mapping.
            logit_single_cloud[single_crop_ind, :] += logit_single_crop
        
        # here finished predictions for a single cloud 
        logit_single_cloud = logit_single_cloud.detach().cpu().numpy()
        pred_single_cloud = np.argmax(logit_single_cloud, axis=1) # prediction: highest score

        # single cloud evaluation 
        area_intersection, area_union, area_target = common_utils.intersectionAndUnion(
            pred_single_cloud.reshape(-1), seg_single_cloud.reshape(-1), 
            len(dataset.class_names), ignore_index=dataset.ignore_index
        )
        class_wise_iou_single_cloud = area_intersection / (area_union + 1e-8)
        mIoU_single_cloud = np.mean(class_wise_iou_single_cloud)
        
        # report metrics for a single cloud
        logger.info(f"\n==== Performance of {region_name}_{sample_idx} ====")
        logger.info(f"{region_name}_{sample_idx}: mIoU: {mIoU_single_cloud}")
        per_cloud_msg.append(f"{region_name}_{sample_idx}: mIoU: {mIoU_single_cloud}")
        for k, c_name in enumerate(dataset.class_names):
            logger.info(f"{c_name}: {class_wise_iou_single_cloud[k]:.4f}")
        logger.info(f"\n====  ====")
        # save this single cloud to the disk  
        save_dict = {
            "points": points,
            "pred": pred_single_cloud,
            "seg": seg_single_cloud,
            "mIoU": mIoU_single_cloud,
            "class_wise_iou": class_wise_iou_single_cloud
        } 
        logger.info(f"Saving {region_name}_{sample_idx}")
        with open(result_dir / f'{region_name}_{sample_idx}.pkl', 'wb') as f:
            pickle.dump(save_dict, f)
        logger.info(f"Saved {region_name}_{sample_idx}")
        # update the global pred and gt lists
        global_pred_list.append(pred_single_cloud)
        global_seg_list.append(seg_single_cloud)
    

    # here finished all clouds
    # start global evaluation 

    global_pred = np.concatenate(global_pred_list)
    global_seg = np.concatenate(global_seg_list)
    area_intersection, area_union, area_target = common_utils.intersectionAndUnion(
        global_pred.reshape(-1), global_seg.reshape(-1), 
        len(dataset.class_names), ignore_index=dataset.ignore_index
    )
    class_wise_iou_global = area_intersection / (area_union + 1e-8)
    mIoU_global = np.mean(class_wise_iou_global)
    
    # report mIoUs for each cloud
    logger.info(f"\n======= Per sample performance ========")
    for msg in per_cloud_msg:
        logger.info(msg)

    # report metrics for a single cloud
    logger.info(f"\n======= Global performance: mIoU: {mIoU_global}========")
    for k, c_name in enumerate(dataset.class_names):
        logger.info(f"{c_name}: {class_wise_iou_global[k]:.4f}")

    logger.info("Saving global predictions and gts")
    save_dict = {
            "pred": global_pred,
            "seg": global_seg,
            "mIoU": mIoU_global,
            "class_wise_iou": class_wise_iou_global
        } 
    with open(result_dir / 'all_results.pkl', 'wb') as f:
        pickle.dump(save_dict, f)

    test_time = time.time() - start_time
    logger.info(f"Testing took {test_time:.2f} second, average model inference time: {model_inference_timer.avg:.2f} second/batch")

    ret_dict = save_dict
    return ret_dict



# def eval_one_epoch_pureforest(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
#     # Latin name to label mapping (with underbars instead of spaces)
#     latin_to_label_underbar = {
#         "Quercus_petraea": "Deciduous_oak",
#         "Quercus_pubescens": "Deciduous_oak",
#         "Quercus_robur": "Deciduous_oak",
#         "Quercus_rubra": "Deciduous_oak",
#         "Quercus_ilex": "Evergreen_oak",
#         "Fagus_sylvatica": "Beech",
#         "Castanea_sativa": "Chestnut",
#         "Robinia_pseudoacacia": "Black_locust",
#         "Pinus_pinaster": "Maritime_pine",
#         "Pinus_sylvestris": "Scotch_pine",
#         "Pinus_nigra_laricio": "Black_pine",
#         "Pinus_nigra": "Black_pine",
#         "Pinus_halepensis": "Aleppo_pine",
#         "Abies_alba": "Fir",
#         "Abies_nordmanniana": "Fir",
#         "Picea_abies": "Spruce",
#         "Larix_decidua": "Larch",
#         "Pseudotsuga_menziesii": "Douglas"
#     }

#     label_to_int = {
#         'Deciduous_oak': 0,
#         'Evergreen_oak': 1,
#         'Beech': 2,
#         'Chestnut': 3,
#         'Black_locust': 4,
#         'Maritime_pine': 5,
#         'Scotch_pine': 6,
#         'Black_pine': 7,
#         'Aleppo_pine': 8,
#         'Fir': 9,
#         'Spruce': 10,
#         'Larch': 11,
#         'Douglas': 12,
#     }

#     # Create a mapping from label to a unique integer starting from 0
#     # label_to_int = {label: i for i, label in enumerate(set(latin_to_label_underbar.values()))}

#     # Function to get the label name by giving the latin name (with underbars)
#     def get_label_from_latin(latin_name):
#         return latin_to_label_underbar.get(latin_name, "Label not found")

#     # Function to get the unique integer ID by giving the label name
#     def get_int_from_label(label_name):
#         return label_to_int.get(label_name, "Label not found")
#     # for classification test 
#     result_dir.mkdir(parents=True, exist_ok=True)

#     final_output_dir = result_dir / 'final_result' / 'data'
#     if args.save_to_file:
#         final_output_dir.mkdir(parents=True, exist_ok=True)

#     dataset = dataloader.dataset
#     class_names = dataset.class_names
#     num_classes = len(class_names)

#     model_inference_timer = common_utils.AverageMeter() # model inference speed

#     logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
#     if dist_test:
#         num_gpus = torch.cuda.device_count()
#         local_rank = cfg.LOCAL_RANK % num_gpus
#         model = torch.nn.parallel.DistributedDataParallel(
#                 model,
#                 device_ids=[local_rank],
#                 broadcast_buffers=False
#         )

#     model.eval()

#     if cfg.LOCAL_RANK == 0:
#         progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    
#     start_time = time.time()
    
#     # load data 
#     dataset = dataloader.dataset # get dataset class
#     num_pc = dataloader.__len__() # all test samples (loop=1)
#     logger.info(f"Total samples to be tested: {num_pc}")
#     global_pred_list = [] # store predictions list(num_test_cloud)
#     global_label_list = [] # store ground truth labels list(num_test_cloud)
    
    
#     # for each cloud
#     pred_save_path = result_dir / 'results.pkl'
#     if exists(pred_save_path): # load the saved prediction
#         logger.info(f"Loaded {pred_save_path}")
#         with open(pred_save_path, 'rb') as f:
#             save_dict = pickle.load(f) # load saved pickles
#             # report the singel crop performance
#             mIoU_single_cloud = save_dict["mIoU"]
#             class_wise_iou_single_cloud = save_dict["class_wise_iou"]
#             logger.info(f"\n==== Performance evaluation ====")
#             logger.info(f"mIoU: {mIoU_single_cloud}")
#             for k, c_name in enumerate(dataset.class_names):
#                 logger.info(f"{c_name}: {class_wise_iou_single_cloud[k]:.4f}")
#             logger.info(f"\n====  ====")
#     else: 
#         batch_size_test = 128 # TODO set by hyperparam
#         data_dict_list = []
#         batch_sample_counter = 0
#         for cloud_id, index in enumerate(range(num_pc)):
#             # load each cloud
#             info = copy.deepcopy(dataset.infos[index])
#             pc_info = info['point_cloud']
#             region_name = pc_info['region_name']
#             sample_idx = pc_info['sample_idx']
#             # cloud_save_path = result_dir / f'{region_name}_{sample_idx}.pkl'
#             pred_save_path = result_dir / 'pred.pkl'
                

#             points = dataset.get_lidar(region_name, sample_idx)
#             points = points[:, :-1] # remove classiifcation
#             latin_name = sample_idx.split('-')[1] # the latin name of tree species 
#             label = get_label_from_latin(latin_name) # latin name from file name
#             label_id = get_int_from_label(label) # converted into unique uid 
#             labels = label_id
            
#             # prepare inputs 
#             input_dict = {
#                 "points": points,
#             }

#             # point feature encoder 
#             data_dict = dataset.point_feature_encoder.forward(input_dict)

#             # data processor
#             data_processor_queue = dataset.data_processor.data_processor_queue
#             for cur_processor in data_processor_queue:
#                 if cur_processor.func.__name__ == 'pillar_crop': # already cropped. 
#                     # print(f"{cur_processor.func.__name__} skipped.")
#                     continue
#                 data_dict = cur_processor(data_dict=data_dict)
                    
#             # remove unnecessary data
#             _ = data_dict.pop('shuffle_idx', None) # because data processing invovles shuffling
#             _ = data_dict.pop('random_sample_indices', None)
#             _ = data_dict.pop('inverse_mapping', None)
            
#             # data_prep_time = time.time() - data_prep_start_time
#             data_dict['labels'] = label_id            
            
#             data_dict_list.append(data_dict)

#             if len(data_dict_list) != batch_size_test: 
#                 batch_sample_counter += 1 
#                 logger.info(f'Accumulating batch samples {batch_sample_counter}')
#             else:# when reached the test batch size, predict once.
#                 logger.info("Start Prediction")
#                 batch_num = int(np.ceil(len(data_dict_list) / batch_size_test)) # TODO: jsut predict, don't put uncecessary for loop, as it predicts only once.
#                 for batch_i in range(batch_num):
#                     # prepare input and model prediction 
#                     s_i, e_i = batch_i * batch_size_test, min((batch_i + 1) * batch_size_test, len(data_dict_list))
#                     data_dict = data_dict_list[s_i:e_i]
#                     # collate
#                     if not isinstance(data_dict, list):
#                         data_dict = [data_dict]
#                     batch_dict = dataset.collate_batch(data_dict)
#                     internal_batch_size = batch_dict["batch_size"]
#                     # convert the batch into torch tensor
#                     load_data_to_gpu(batch_dict)# in "model_func_decorator"

#                     # with torch.cuda.amp.autocast(enabled=use_amp):
#                     model.eval()
#                     model_forward_start = time.time()
#                     with torch.no_grad():
#                         ret_dict, tb_dict, disp_dict = model(batch_dict)
#                     model_forward_time = time.time() - model_forward_start
#                     model_inference_timer.update(model_forward_time/internal_batch_size)
                    
#                     # just for reference to see if scores are normal
#                     batch_mIoU = ret_dict["batch_mIoU"]
#                     loss = ret_dict["loss"]

#                     logits = disp_dict['logits']
#                     labels = disp_dict['labels']
#                     global_pred_list.append(logits.detach().cpu().numpy())
#                     global_label_list.append(labels.detach().cpu().numpy())

#                     # logger.info(f"  batch mIoU: {batch_mIoU:.4f}, loss: {loss:.4f}, model speed: {model_inference_timer.avg:.2f} sec/batch")
#                     logger.info(f"batch mIoU: {batch_mIoU:.4f}, loss: {loss:.4f}, model speed: {model_inference_timer.avg:.2f} sec/batch \n \
#                                 {cloud_id+1}/{num_pc}")
                    
#                     # reset
#                     data_dict_list = []
#                     batch_sample_counter = 0
            
            
#             # if (cloud_id+1) > 2 * batch_size_test:# debug
#             #     break 

#         # here finished all cloud prediction
#         # start global evaluation 

#         global_preds = np.concatenate(global_pred_list)
#         global_preds = np.argmax(global_preds, 1)
#         global_labels = np.concatenate(global_label_list)
#         area_intersection, area_union, area_target = common_utils.intersectionAndUnion(
#             global_preds.reshape(-1), global_labels.reshape(-1), 
#             len(dataset.class_names), ignore_index=dataset.ignore_index
#         )
#         class_wise_iou_global = area_intersection / (area_union + 1e-8)
#         mIoU_global = np.mean(class_wise_iou_global)

#         # report metrics for a single cloud
#         logger.info(f"\n======= Global performance: mIoU: {mIoU_global}========")
#         for k, c_name in enumerate(dataset.class_names):
#             logger.info(f"{c_name}: {class_wise_iou_global[k]:.4f}")

#         logger.info("Saving global predictions and gts")
#         save_dict = {
#                 "preds": global_preds,
#                 "labels": global_labels,
#                 "mIoU": mIoU_global,
#                 "class_wise_iou": class_wise_iou_global
#             } 
#         with open(result_dir / 'results.pkl', 'wb') as f:
#             pickle.dump(save_dict, f)

#         test_time = time.time() - start_time
#         logger.info(f"Testing took {test_time:.2f} second, average model inference time: {model_inference_timer.avg:.2f} second/batch")

#     ret_dict = save_dict
#     return ret_dict

def eval_one_epoch_pureforest(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    # Latin name to label mapping (with underbars instead of spaces)
    latin_to_label_underbar = {
        "Quercus_petraea": "Deciduous_oak",
        "Quercus_pubescens": "Deciduous_oak",
        "Quercus_robur": "Deciduous_oak",
        "Quercus_rubra": "Deciduous_oak",
        "Quercus_ilex": "Evergreen_oak",
        "Fagus_sylvatica": "Beech",
        "Castanea_sativa": "Chestnut",
        "Robinia_pseudoacacia": "Black_locust",
        "Pinus_pinaster": "Maritime_pine",
        "Pinus_sylvestris": "Scotch_pine",
        "Pinus_nigra_laricio": "Black_pine",
        "Pinus_nigra": "Black_pine",
        "Pinus_halepensis": "Aleppo_pine",
        "Abies_alba": "Fir",
        "Abies_nordmanniana": "Fir",
        "Picea_abies": "Spruce",
        "Larix_decidua": "Larch",
        "Pseudotsuga_menziesii": "Douglas"
    }

    label_to_int = {
        'Deciduous_oak': 0,
        'Evergreen_oak': 1,
        'Beech': 2,
        'Chestnut': 3,
        'Black_locust': 4,
        'Maritime_pine': 5,
        'Scotch_pine': 6,
        'Black_pine': 7,
        'Aleppo_pine': 8,
        'Fir': 9,
        'Spruce': 10,
        'Larch': 11,
        'Douglas': 12,
    }

    # Create a mapping from label to a unique integer starting from 0
    # label_to_int = {label: i for i, label in enumerate(set(latin_to_label_underbar.values()))}

    # Function to get the label name by giving the latin name (with underbars)
    def get_label_from_latin(latin_name):
        return latin_to_label_underbar.get(latin_name, "Label not found")

    # Function to get the unique integer ID by giving the label name
    def get_int_from_label(label_name):
        return label_to_int.get(label_name, "Label not found")
    # for classification test 
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset
    class_names = dataset.class_names
    num_classes = len(class_names)

    model_inference_timer = common_utils.AverageMeter() # model inference speed

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )

    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    
    start_time = time.time()
    
    # load data 
    dataset = dataloader.dataset # get dataset class
    num_pc = dataloader.__len__() # all test samples (loop=1)
    # logger.info(f"Total samples to be tested: {num_pc}")
    global_pred_list = [] # store predictions list(num_test_cloud)
    global_label_list = [] # store ground truth labels list(num_test_cloud)
    
    
    # for each cloud
    pred_save_path = result_dir / 'results.pkl'
    if exists(pred_save_path): # load the saved prediction
        logger.info(f"Loaded {pred_save_path}")
        with open(pred_save_path, 'rb') as f:
            save_dict = pickle.load(f) # load saved pickles
            # report the singel crop performance
            mIoU_single_cloud = save_dict["mIoU"]
            class_wise_iou_single_cloud = save_dict["class_wise_iou"]
            logger.info(f"\n==== Performance evaluation ====")
            logger.info(f"mIoU: {mIoU_single_cloud}")
            for k, c_name in enumerate(dataset.class_names):
                logger.info(f"{c_name}: {class_wise_iou_single_cloud[k]:.4f}")
            logger.info(f"\n====  ====")
    else: 
        # for cloud_id, index in enumerate(range(num_pc)):
        for batch_id, batch_dict in enumerate(dataloader):
            # prepare input and model prediction 
            load_data_to_gpu(batch_dict)
            model.eval()
            model_forward_start = time.time()
            with torch.no_grad():
                ret_dict, tb_dict, disp_dict = model(batch_dict)
            model_forward_time = time.time() - model_forward_start
            model_inference_timer.update(model_forward_time/batch_dict['batch_size'])
            
            # just for reference to see if scores are normal
            batch_mIoU = ret_dict["batch_mIoU"]
            loss = ret_dict["loss"]

            logits = disp_dict['logits']
            labels = disp_dict['labels']
            global_pred_list.append(logits.detach().cpu().numpy())
            global_label_list.append(labels.detach().cpu().numpy())

            # logger.info(f"  batch mIoU: {batch_mIoU:.4f}, loss: {loss:.4f}, model speed: {model_inference_timer.avg:.2f} sec/batch")
            logger.info(f"batch mIoU: {batch_mIoU:.4f}, loss: {loss:.4f}, model speed: {model_inference_timer.avg:.2f} sec/batch \n \
                        {batch_id+1}/{num_pc}")
            
        
            
            
            # break 

        # here finished all cloud prediction
        # start global evaluation 

        global_preds = np.concatenate(global_pred_list)
        global_preds = np.argmax(global_preds, 1)
        global_labels = np.concatenate(global_label_list)
        area_intersection, area_union, area_target = common_utils.intersectionAndUnion(
            global_preds.reshape(-1), global_labels.reshape(-1), 
            len(dataset.class_names), ignore_index=dataset.ignore_index
        )
        class_wise_iou_global = area_intersection / (area_union + 1e-8)
        mIoU_global = np.mean(class_wise_iou_global)

        # report metrics for a single cloud
        logger.info(f"\n======= Global performance: mIoU: {mIoU_global}========")
        for k, c_name in enumerate(dataset.class_names):
            logger.info(f"{c_name}: {class_wise_iou_global[k]:.4f}")

        logger.info("Saving global predictions and gts")
        save_dict = {
                "preds": global_preds,
                "labels": global_labels,
                "mIoU": mIoU_global,
                "class_wise_iou": class_wise_iou_global
            } 
        with open(result_dir / 'results.pkl', 'wb') as f:
            pickle.dump(save_dict, f)

        test_time = time.time() - start_time
        logger.info(f"Testing took {test_time:.2f} second, average model inference time: {model_inference_timer.avg:.2f} second/batch")

    ret_dict = save_dict
    return ret_dict

def eval_one_epoch_opengfcls(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    
    # for classification test 
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset
    class_names = dataset.class_names
    num_classes = len(class_names)

    model_inference_timer = common_utils.AverageMeter() # model inference speed

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )

    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    
    start_time = time.time()
    
    # load data 
    dataset = dataloader.dataset # get dataset class
    num_pc = dataloader.__len__() # all test samples (loop=1)
    # logger.info(f"Total samples to be tested: {num_pc}")
    global_pred_list = [] # store predictions list(num_test_cloud)
    global_label_list = [] # store ground truth labels list(num_test_cloud)
    
    
    # for each cloud
    pred_save_path = result_dir / 'results.pkl'
    if exists(pred_save_path): # load the saved prediction
        logger.info(f"Loaded {pred_save_path}")
        with open(pred_save_path, 'rb') as f:
            save_dict = pickle.load(f) # load saved pickles
            # report the singel crop performance
            mIoU_single_cloud = save_dict["mIoU"]
            OA_single_cloud = save_dict["OA"]
            class_wise_iou_single_cloud = save_dict["class_wise_iou"]
            logger.info(f"\n==== Performance evaluation ====")
            logger.info(f"mIoU: {mIoU_single_cloud}")
            logger.info(f"OA: {OA_single_cloud}")
            for k, c_name in enumerate(dataset.class_names):
                logger.info(f"{c_name}: {class_wise_iou_single_cloud[k]:.4f}")
            logger.info(f"\n====  ====")
    else: 
        # for cloud_id, index in enumerate(range(num_pc)):
        for batch_id, batch_dict in enumerate(dataloader):
            # prepare input and model prediction 
            load_data_to_gpu(batch_dict)
            model.eval()
            model_forward_start = time.time()
            with torch.no_grad():
                ret_dict, tb_dict, disp_dict = model(batch_dict)
            model_forward_time = time.time() - model_forward_start
            model_inference_timer.update(model_forward_time/batch_dict['batch_size'])
            
            # just for reference to see if scores are normal
            batch_mIoU = ret_dict["batch_mIoU"]
            loss = ret_dict["loss"]

            logits = disp_dict['logits']
            labels = disp_dict['labels']
            global_pred_list.append(logits.detach().cpu().numpy())
            global_label_list.append(labels.detach().cpu().numpy())

            # logger.info(f"  batch mIoU: {batch_mIoU:.4f}, loss: {loss:.4f}, model speed: {model_inference_timer.avg:.2f} sec/batch")
            logger.info(f"batch mIoU: {batch_mIoU:.4f}, loss: {loss:.4f}, model speed: {model_inference_timer.avg:.2f} sec/batch \n \
                        {batch_id+1}/{num_pc}")
            
        
            
            
            # break 

        # here finished all cloud prediction
        # start global evaluation 

        global_preds = np.concatenate(global_pred_list)
        global_preds = np.argmax(global_preds, 1)
        global_labels = np.concatenate(global_label_list)
        area_intersection, area_union, area_target = common_utils.intersectionAndUnion(
            global_preds.reshape(-1), global_labels.reshape(-1), 
            len(dataset.class_names), ignore_index=dataset.ignore_index
        )
        class_wise_iou_global = area_intersection / (area_union + 1e-8)
        mIoU_global = np.mean(class_wise_iou_global)
        overall_acc = np.sum(area_intersection) / np.sum(area_target + 1e-8)
        # report metrics for a single cloud
        logger.info(f"\n======= Global performance: mIoU: {mIoU_global}, OA: {overall_acc}========")
        for k, c_name in enumerate(dataset.class_names):
            logger.info(f"{c_name}: {class_wise_iou_global[k]:.4f}")

        logger.info("Saving global predictions and gts")
        save_dict = {
                "preds": global_preds,
                "labels": global_labels,
                "mIoU": mIoU_global,
                "OA": overall_acc,
                "class_wise_iou": class_wise_iou_global
            } 
        with open(result_dir / 'results.pkl', 'wb') as f:
            pickle.dump(save_dict, f)

        test_time = time.time() - start_time
        logger.info(f"Testing took {test_time:.2f} second, average model inference time: {model_inference_timer.avg:.2f} second/batch")

    ret_dict = save_dict
    return ret_dict


def eval_one_epoch_seg_parallel_prep_v2(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    # this is the version which the model perform inverse mapping by itself
    # PointToVoxel is adopted for transforming points to voxels.
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset
    class_names = dataset.class_names
    num_classes = len(class_names)

    model_inference_timer = common_utils.AverageMeter() # model inference speed

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )

    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    
    start_time = time.time()
    
    # load data 
    dataset = dataloader.dataset # get dataset class
    num_pc = dataloader.__len__() # all test samples (loop=1)
    logger.info(f"Total samples to be tested: {num_pc}")
    global_pred_list = [] # store predictions list(num_test_cloud)
    global_seg_list = [] # store ground truth labels list(num_test_cloud)
    
    
    # for each cloud
    for cloud_id, index in enumerate(range(num_pc)):
        # load each cloud
        info = copy.deepcopy(dataset.infos[index])
        pc_info = info['point_cloud']
        region_name = pc_info['region_name']
        sample_idx = pc_info['sample_idx']
        cloud_save_path = result_dir / f'{region_name}_{sample_idx}.pkl'
        if exists(cloud_save_path): # load the saved prediction
            logger.info(f"Loaded {cloud_save_path}")
            with open(cloud_save_path, 'rb') as f:
                save_dict = pickle.load(f) # load saved pickles
            # report the singel crop performance
            mIoU_single_cloud = save_dict["mIoU"]
            class_wise_iou_single_cloud = save_dict["class_wise_iou"]
            logger.info(f"\n==== Performance of {region_name}_{sample_idx} ====")
            logger.info(f"{region_name}_{sample_idx}: mIoU: {mIoU_single_cloud}")
            for k, c_name in enumerate(dataset.class_names):
                logger.info(f"{c_name}: {class_wise_iou_single_cloud[k]:.4f}")
            logger.info(f"\n====  ====")
            pred_single_cloud = save_dict["pred"]
            seg_single_cloud = save_dict["seg"]
            # update global pred and seg
            global_pred_list.append(pred_single_cloud)
            global_seg_list.append(seg_single_cloud)
            
            continue

        points = dataset.get_lidar(region_name, sample_idx)
        seg_single_cloud = points[:, -1]
        
        # generate crops from original big clouds. Each crop size = pillar size 
        # crop_size = dataset.data_processor.processor_configs.pillar_crop.PILLAR_SIZE
        crop_size= 256 # TODO: load from config pillar crop size
        stride = crop_size//4  # the amount of overlap
        crop_start_time = time.time()
        crop_list, crop_ind_list = generate_crops(points, crop_size, stride)
        crop_time = time.time() - crop_start_time
        flattened_crop_ind_list = [item for sublist in crop_ind_list for item in sublist]
        assert len(np.unique(flattened_crop_ind_list)) == len(points), f"the number of unique indices is {len(np.unique(flattened_crop_ind_list))}, while should have {len(points)}"  # make sure no points missing.
        logger.info(f"Number of crops for cloud {region_name}_{sample_idx}: {len(crop_ind_list)}. Crop size {crop_size}, stride: {stride}, took {crop_time:.2f} seconds for cropping.")

        # for each crop 
        logit_single_cloud = torch.zeros((points.shape[0], len(dataset.class_names))).cuda() # container for predictions on this single crop
        for crop_id, (single_crop, single_crop_ind) in enumerate(zip(crop_list, crop_ind_list)):
            # generate idx_data that covers all 
            coord = single_crop[:, :3]
            feat = single_crop[:, 3:-1]
            label = single_crop[:, -1]
            idx_data = []
            coord_min = np.min(coord, 0)
            coord = coord - coord_min
            
            # the voxel size for each sample. 
            # Should be smaller than traiing voxel size so that denser points can be obtained 
            # voxel_size_train = dataset.data_processor.processor_configs.transform_points_to_voxels_numpy.VOXEL_SIZE
            # voxel_size = voxel_size_train * 0.2 
            voxel_size = 0.2 # TODO load from config
            voxelize_start_time = time.time()
            idx_sort, count = voxelize(coord, voxel_size, mode=1) # validation mode 
            voxelize_time = time.time() - voxelize_start_time
            for i in range(count.max()): # ensure all points within a voxle are taken at least once
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                idx_data.append(idx_part)
            
            logger.info(f"There are {len(idx_data)} partial crops for crop {crop_id+1}/{len(crop_list)}. Voxelizations took {voxelize_time:.2f} sconds")
            
            # each element in idx_data is a cropped tile voxelized by voxel_size. 
            # pointcept like sphere sampling is not implemented furhter. 
            # will use inverse mapping to obtain the full predictions for crops

            # for each partial crop, prepare data and create batches 
            logit_single_crop = torch.zeros((label.size, len(dataset.class_names))).cuda() # container for predictions on this single crop
            
            # accumulate prepareed data of a partial crop 
            logger.info(f"Start preparing data for crop {crop_id+1}/{len(crop_list)}")
            data_dict_list = []
            data_prep_start_time  = time.time()

            

            def parallel_processing_multiprocessing(idx_data, single_crop, dataset, num_workers=None):
                data_dict_list = []
                
                # Prepare the argument list for each process
                args_list = [(partial_crop_id, partial_index, idx_data, single_crop, dataset) 
                            for partial_crop_id, partial_index in enumerate(idx_data)]
                
                # Use multiprocessing Pool
                with multiprocessing.Pool(processes=num_workers, maxtasksperchild=1) as pool:
                    # Distribute the tasks across workers
                    results = pool.map(process_partial_crop_v2, args_list)
                
                # Collect results
                data_dict_list.extend(results)
                
                return data_dict_list

            data_dict_list = parallel_processing_multiprocessing(idx_data, single_crop, dataset, num_workers=3)

            data_prep_time = time.time() - data_prep_start_time

            logger.info(f"Start prediction for the crop {crop_id+1}/{len(crop_list)}. Preparation took {data_prep_time:.2f} seconds.")
            batch_size_test = 16 # TODO set by hyperparam
            batch_num = int(np.ceil(len(data_dict_list) / batch_size_test))
            for batch_i in range(batch_num):
                # prepare input and model prediction 
                s_i, e_i = batch_i * batch_size_test, min((batch_i + 1) * batch_size_test, len(data_dict_list))
                data_dict = data_dict_list[s_i:e_i]
                partial_indices = idx_data[s_i:e_i]
                # collate
                if not isinstance(data_dict, list):
                    data_dict = [data_dict]
                    partial_indices = [partial_indices]
                batch_dict = dataset.collate_batch(data_dict)
                internal_batch_size = batch_dict["batch_size"]
                # convert the batch into torch tensor
                load_data_to_gpu(batch_dict)# in "model_func_decorator"

                # with torch.cuda.amp.autocast(enabled=use_amp):
                model.eval()
                model_forward_start = time.time()
                with torch.no_grad():
                   ret_dict, tb_dict, disp_dict = model(batch_dict)
                model_forward_time = time.time() - model_forward_start
                model_inference_timer.update(model_forward_time/internal_batch_size)
                
                # just for reference to see if scores are normal
                batch_mIoU = ret_dict["batch_mIoU"]
                loss = ret_dict["loss"]
                
                # handle predictions
                internal_batch_size = batch_dict["batch_size"]
                # internal_batch_idx_sampled =  batch_dict["voxel_coords"][:, 0] # prediction
                internal_batch_idx_full =  batch_dict["points"][:, 0] # inverse mapping, shuffle_inverse_idx
                for b_i in range(internal_batch_size):
                    # recover partial crop in its original order by interating the batch
                    # import pdb; pdb.set_trace()
                    selected_full = internal_batch_idx_full == b_i
                    
                    sampled_partial_logits = disp_dict["pred"][selected_full] # logits
                    shuffle_idx_inverse = batch_dict["shuffle_idx_inverse"][selected_full]
                    partial_logits = sampled_partial_logits[shuffle_idx_inverse] # unshuffle

                    # add to the prediction 
                    partial_index = partial_indices[b_i]
                    logit_single_crop[partial_index, :] += partial_logits
                # logger.info(f"  batch mIoU: {batch_mIoU:.4f}, loss: {loss:.4f}, model speed: {model_inference_timer.avg:.2f} sec/batch")
                logger.info(f"batch mIoU: {batch_mIoU:.4f}, loss: {loss:.4f}, model speed: {model_inference_timer.avg:.2f} sec/batch\n \
                            partial crop, crop, cloud: {e_i}/{len(idx_data)}, {crop_id+1}/{len(crop_list)}, {cloud_id+1}/{num_pc}")
            # here finished for a single crop 
            # now assign predictions to the original cloud by crop --> original cloud mapping.
            logit_single_cloud[single_crop_ind, :] += logit_single_crop
        
        # here finished predictions for a single cloud 
        logit_single_cloud = logit_single_cloud.detach().cpu().numpy()
        pred_single_cloud = np.argmax(logit_single_cloud, axis=1) # prediction: highest score

        # single cloud evaluation 
        area_intersection, area_union, area_target = common_utils.intersectionAndUnion(
            pred_single_cloud.reshape(-1), seg_single_cloud.reshape(-1), 
            len(dataset.class_names), ignore_index=dataset.ignore_index
        )
        class_wise_iou_single_cloud = area_intersection / (area_union + 1e-8)
        mIoU_single_cloud = np.mean(class_wise_iou_single_cloud)
        
        # report metrics for a single cloud
        logger.info(f"\n==== Performance of {region_name}_{sample_idx} ====")
        logger.info(f"{region_name}_{sample_idx}: mIoU: {mIoU_single_cloud}")
        for k, c_name in enumerate(dataset.class_names):
            logger.info(f"{c_name}: {class_wise_iou_single_cloud[k]:.4f}")
        logger.info(f"\n====  ====")
        # save this single cloud to the disk  
        save_dict = {
            "points": batch_dict["points"].detach().cpu().numpy(),
            "pred": pred_single_cloud,
            "seg": seg_single_cloud,
            "mIoU": mIoU_single_cloud,
            "class_wise_iou": class_wise_iou_single_cloud
        } 
        logger.info(f"Saving {region_name}_{sample_idx}")
        with open(result_dir / f'{region_name}_{sample_idx}.pkl', 'wb') as f:
            pickle.dump(save_dict, f)
        
        # update the global pred and gt lists
        global_pred_list.append(pred_single_cloud)
        global_seg_list.append(seg_single_cloud)
    

    # here finished all clouds
    # start global evaluation 

    global_pred = np.concatenate(global_pred_list)
    global_seg = np.concatenate(global_seg_list)
    area_intersection, area_union, area_target = common_utils.intersectionAndUnion(
        global_pred.reshape(-1), global_seg.reshape(-1), 
        len(dataset.class_names), ignore_index=dataset.ignore_index
    )
    class_wise_iou_global = area_intersection / (area_union + 1e-8)
    mIoU_global = np.mean(class_wise_iou_global)
    
    # report metrics for a single cloud
    logger.info(f"\n======= Global performance: mIoU: {mIoU_global}========")
    for k, c_name in enumerate(dataset.class_names):
        logger.info(f"{c_name}: {class_wise_iou_global[k]:.4f}")

    logger.info("Saving global predictions and gts")
    save_dict = {
            "pred": global_pred,
            "seg": global_seg,
            "mIoU": mIoU_global,
            "class_wise_iou": class_wise_iou_global
        } 
    with open(result_dir / 'all_results.pkl', 'wb') as f:
        pickle.dump(save_dict, f)

    test_time = time.time() - start_time
    logger.info(f"Testing took {test_time:.2f} second, average model inference time: {model_inference_timer.avg:.2f} second/batch")

    ret_dict = save_dict
    return ret_dict

def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr

def voxelize(coord, voxel_size=0.05, hash_type='fnv', mode=0):
    discrete_coord = np.floor(coord / np.array(voxel_size))
    
    key = fnv_hash_vec(discrete_coord)

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, count = np.unique(key_sort, return_counts=True)
    if mode == 0:  # train mode
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
        idx_unique = idx_sort[idx_select]
        return idx_unique
    else:  # val mode
        return idx_sort, count

def generate_crops(point_cloud, crop_size, stride):
    """
    Split a 3D point cloud into crops of a specified size with a given stride.
    
    Parameters:
    point_cloud (numpy.ndarray): A Nx3 array representing the point cloud, where each row is a point [x, y, z].
    crop_size (float): The size of each crop along the x and y dimensions.
    stride (float): The stride size for moving the crop window, allowing for overlapping crops.
    
    Returns:
    list: A list of numpy arrays, where each array represents a cropped point cloud.
    list: A list of boolean arrays indicating the indices of points in the original point cloud for each crop.
    """
    # Determine the range of x and y coordinates
    min_x, min_y = np.min(point_cloud[:, :2], axis=0)
    max_x, max_y = np.max(point_cloud[:, :2], axis=0)
    
    crops = []
    crop_inds = []

    # Generate crops by iterating over the x and y ranges with the specified stride
    x_start = min_x
    while x_start < max_x:
        y_start = min_y
        while y_start < max_y:
            # Define the current crop boundaries
            if x_start + crop_size < max_x:
                x_start_crop = x_start
                x_end_crop = x_start + crop_size
            else:
                x_start_crop = max_x - crop_size
                x_end_crop = max_x
            
            if y_start + crop_size < max_y:
                y_start_crop = y_start
                y_end_crop = y_start + crop_size
            else:
                y_start_crop = max_y - crop_size
                y_end_crop = max_y

            # Extract points that fall within the current crop boundaries (& seems to be faster than np.logical_and)
            crop_ind = (point_cloud[:, 0] >= x_start_crop) & (point_cloud[:, 0] <= x_end_crop) & \
                    (point_cloud[:, 1] >= y_start_crop) & (point_cloud[:, 1] <= y_end_crop)
            # crop_ind = np.logical_and(
            #     np.logical_and(point_cloud[:, 0] >= x_start_crop, point_cloud[:, 0] <= x_end_crop),
            #     np.logical_and(point_cloud[:, 1] >= y_start_crop, point_cloud[:, 1] <= y_end_crop)
            # )

            crop = point_cloud[crop_ind]
            # if it's not empty
            if crop.size > 0: 
                # move crop to the cropping center. make min_z = 0
                translation_x = x_start_crop + (x_end_crop - x_start_crop)/2
                translation_y = y_start_crop + (y_end_crop - y_start_crop)/2
                translation_z = np.min(crop[:, 2])
                crop[:, :3] = crop[:, :3] - np.array([translation_x, translation_y, translation_z])
                # import pdb; pdb.set_trace()
            
                # Add the crop to the list of crops 
                crops.append(crop)
                crop_inds.append(np.arange(point_cloud.shape[0])[crop_ind])
            
            if y_end_crop == max_y:
                # if has reached the boundary
                break
            else:
                # Move to the next y position with stride
                y_start += stride
        
        if x_end_crop == max_x:
            # if has reached the boundary
            break
        else:
            # Move to the next x position with stride
            x_start += stride

    return crops, crop_inds
        


def process_partial_crop(args):
    partial_crop_id, partial_index, idx_data, single_crop, dataset = args
    
    partial_crop = single_crop[partial_index]
    partial_points = partial_crop[:, :-1]
    partial_seg = partial_crop[:, -1]
    input_dict = {
        "points": partial_points,
        "point_seg_labels_full": partial_seg
    }
    
    # Prepare data (emulate __getitem__ in dataset)
    
    # Point feature encoder
    data_dict = dataset.point_feature_encoder.forward(input_dict)
    
    # Data processor
    data_processor_queue = dataset.data_processor.data_processor_queue
    for cur_processor in data_processor_queue:
        if cur_processor.func.__name__ == 'pillar_crop':  # already cropped
            continue
        data_dict = cur_processor(data_dict=data_dict)
    
    # Get shuffle and inverse mapping such that the original order can be recovered
    shuffle_idx = data_dict.pop('shuffle_idx', None)  # because data processing involves shuffling
    shuffle_idx_inverse = np.argsort(shuffle_idx)  # to recover the original order later
    data_dict['shuffle_idx_inverse'] = shuffle_idx_inverse
    
    random_sample_idx = data_dict.pop('random_sample_indices', None)
    sampled_partial_seg = partial_seg[shuffle_idx] if shuffle_idx is not None else partial_seg
    sampled_partial_seg = sampled_partial_seg[random_sample_idx] if random_sample_idx is not None else sampled_partial_seg
    sampled_indices = data_dict['sampled_indices']
    sampled_partial_seg_grouped = sampled_partial_seg[sampled_indices]
    sampled_partial_seg_aggregated = mode(sampled_partial_seg_grouped, axis=1, keepdims=False).mode
    data_dict['point_seg_labels'] = sampled_partial_seg_aggregated.reshape(-1, 1)
    
    return data_dict

def process_partial_crop_v2(args):
    # points to voxel uses PointToVoxel
    partial_crop_id, partial_index, idx_data, single_crop, dataset = args
    
    partial_crop = single_crop[partial_index]
    partial_points = partial_crop[:, :-1]
    partial_seg = partial_crop[:, -1]
    input_dict = {
        "points": partial_points,
        "point_seg_labels_full": partial_seg
    }
    
    # Prepare data (emulate __getitem__ in dataset)
    
    # Point feature encoder
    data_dict = dataset.point_feature_encoder.forward(input_dict)
    
    # Data processor
    data_processor_queue = dataset.data_processor.data_processor_queue
    for cur_processor in data_processor_queue:
        print(cur_processor.func.__name__, flush=True)
        if cur_processor.func.__name__ == 'pillar_crop':  # already cropped
            continue
        data_dict = cur_processor(data_dict=data_dict)
    
    # Get shuffle and inverse mapping such that the original order can be recovered
    shuffle_idx = data_dict.pop('shuffle_idx', None)  # because data processing involves shuffling
    shuffle_idx_inverse = np.argsort(shuffle_idx)  # to recover the original order later
    data_dict['shuffle_idx_inverse'] = shuffle_idx_inverse
    
    random_sample_idx = data_dict.pop('random_sample_indices', None)
    sampled_partial_seg = partial_seg[shuffle_idx] if shuffle_idx is not None else partial_seg
    sampled_partial_seg = sampled_partial_seg[random_sample_idx] if random_sample_idx is not None else sampled_partial_seg
    inverse_mapping = data_dict.get('inverse_mapping', None)
    if inverse_mapping is not None:
        valid_mask = inverse_mapping != -1 # this shoud not contain -1
        # assert np.all(valid_mask), "mask contains invalid points. This should not happen when test."
        sampled_partial_seg = sampled_partial_seg[valid_mask]
    # sampled_indices = data_dict['sampled_indices']
    # sampled_partial_seg_grouped = sampled_partial_seg[sampled_indices]
    # sampled_partial_seg_aggregated = mode(sampled_partial_seg_grouped, axis=1, keepdims=False).mode
    data_dict['point_seg_labels'] = sampled_partial_seg.reshape(-1, 1)
    
    return data_dict

if __name__ == '__main__':
    pass
