import logging
import os
import pickle
import random
import shutil
import subprocess
import SharedArray

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def random_masking(N, L, mask_ratio, device):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    N: 1, L: number of voxels. Voxel-wise masking. 
    """
    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(N, L, device=device)  # (1, num_voxel), noise in [0, 1]
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=device) # (1, num_voxel)
    mask.scatter_(1, ids_keep, 0)
    return mask

def get_in_range_mask(points, pc_range, voxel_size, grid_size):
    if not isinstance(pc_range, torch.Tensor):
        pc_range = points.new_tensor(pc_range)
    if not isinstance(voxel_size, torch.Tensor):
        voxel_size = points.new_tensor(voxel_size)
    if not isinstance(grid_size, torch.Tensor):
        grid_size = points.new_tensor(grid_size).to(torch.int64)
    
    # generate voxel grids 
    coords = ((points[:, 1:4] - pc_range[:3]) / voxel_size).to(torch.int64) # 1st dim of points is batch idx
    mask = torch.all((coords[:, :3] >= grid_size.new_zeros(grid_size.shape)) & (coords[:, :3] < grid_size), dim=-1) # check if points lie in the pre-defined (preprocessor) grid size
    return mask, coords


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape, f"output shape: {output.shape} != target shape {target.shape}"
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    # assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    assert output.shape == target.shape
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def is_main_process():
    """
    Checks if the current process is the main one (rank 0).
    """
    return dist.get_rank() == 0

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def limit_period(val, offset=0.5, period=np.pi):
    val, is_numpy = check_numpy_to_torch(val)
    ans = val - torch.floor(val / period + offset) * period
    return ans.numpy() if is_numpy else ans


def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def angle2matrix(angle):
    """
    Args:
        angle: angle along z-axis, angle increases x ==> y
    Returns:
        rot_matrix: (3x3 Tensor) rotation matrix
    """

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    rot_matrix = torch.tensor([
        [cosa, -sina, 0],
        [sina, cosa,  0],
        [   0,    0,  1]
    ])
    return rot_matrix


def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4]) \
           & (points[:, 2] >= limit_range[2]) & (points[:, 2] <= limit_range[5])
    return mask


def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers


def get_voxel_centers_xy(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    voxel_centers[:, 2] = 0 # let z center to be 0. No translation.
    return voxel_centers


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, seed=777):
    if seed is not None:
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
        torch.cuda.manual_seed(seed + worker_id)
        torch.cuda.manual_seed_all(seed + worker_id)


def get_pad_params(desired_size, cur_size):
    """
    Get padding parameters for np.pad function
    Args:
        desired_size: int, Desired padded output size
        cur_size: int, Current size. Should always be less than or equal to cur_size
    Returns:
        pad_params: tuple(int), Number of values padded to the edges (before, after)
    """
    assert desired_size >= cur_size

    # Calculate amount to pad
    diff = desired_size - cur_size
    pad_params = (0, diff)

    return pad_params


def keep_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


def init_dist_slurm(tcp_port, local_rank, backend='nccl'):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        tcp_port:
        backend:

    Returns:

    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    rank = dist.get_rank()
    return total_gpus, rank


def init_dist_pytorch(tcp_port, local_rank, backend='nccl', host_file_path=None):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    # os.environ['MASTER_PORT'] = str(tcp_port)
    # os.environ['MASTER_ADDR'] = 'localhost'
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        # init_method='tcp://127.0.0.1:%d' % tcp_port,
        # rank=local_rank,
        # world_size=num_gpus
    )
    rank = dist.get_rank()
    return num_gpus, rank

def init_dist_abci(tcp_port, local_rank, backend='nccl', host_file_path=None):
    # os.environ['MASTER_PORT'] = str(tcp_port)
    # os.environ['MASTER_ADDR'] = 'localhost'
    num_gpus = torch.cuda.device_count() # number of device on one machine 
    node_rank = int(os.environ["OMPI_COMM_WORLD_RANK"]) # global rank of a process (GPU)
    torch.cuda.set_device(node_rank % num_gpus) # local rank
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
    print(f"Obtain Host lists from {host_file_path}")
    with open(host_file_path, mode="r") as f:
    # with open('./hostfile', mode="r") as f:
        host = f.readlines()
    host[0] = host[0].rstrip("\n")
    # print(host)
    os.environ["MASTER_ADDR"] = host[0]
    os.environ["MASTER_PORT"] = str(tcp_port)
    dist.init_process_group(
        backend=backend,
        # init_method='tcp://127.0.0.1:%d' % tcp_port,
        # rank=local_rank,
        # world_size=num_gpus
    )
    rank = dist.get_rank()
    return num_gpus, rank

# TODO multi-node training for ABCI
# def init_dist_pytorch_abci(tcp_port, local_rank, backend='nccl'):
#     if mp.get_start_method(allow_none=True) is None:
#         mp.set_start_method('spawn')
    
#     # get abci host information
#     current_dir = os.getcwd()
#     with open(current_dir + "/hostfile") as f:
#         host = f.readlines()
#     host[0] = host[0].rstrip("\n")
#     dist_url = "tcp://" + host[0] + ":" + str(tcp_port)
    
#     node_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])  # Process number in MPI
#     size = int(os.environ["OMPI_COMM_WORLD_SIZE"])  # The all size of process
#     print("node rank:{}".format(node_rank))
#     print("size of process:{}".format(size))
#     num_gpus = torch.cuda.device_count()  # gpu num per node
#     world_size = gpu * size  # total gpu num
#     print(world_size)

#     # num_gpus = torch.cuda.device_count()
#     # torch.cuda.set_device(local_rank % num_gpus)

#     dist.init_process_group(
#         backend=backend,
#         init_method=dist_url,
#         # rank=local_rank,
#         # world_size=num_gpus
#     )
#     rank = dist.get_rank()
#     return num_gpus, rank

def get_dist_info(return_gpu_per_machine=False):
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if return_gpu_per_machine:
        gpu_per_machine = torch.cuda.device_count()
        return rank, world_size, gpu_per_machine

    return rank, world_size


def merge_results_dist(result_part, size, tmpdir):
    rank, world_size = get_dist_info()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(result_part, open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
    dist.barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
        part_list.append(pickle.load(open(part_file, 'rb')))

    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results


def scatter_point_inds(indices, point_inds, shape):
    ret = -1 * torch.ones(*shape, dtype=point_inds.dtype, device=point_inds.device)
    ndim = indices.shape[-1]
    flattened_indices = indices.view(-1, ndim)
    slices = [flattened_indices[:, i] for i in range(ndim)]
    ret[slices] = point_inds
    return ret


def generate_voxel2pinds(sparse_tensor):
    device = sparse_tensor.indices.device
    batch_size = sparse_tensor.batch_size
    spatial_shape = sparse_tensor.spatial_shape
    indices = sparse_tensor.indices.long()
    point_indices = torch.arange(indices.shape[0], device=device, dtype=torch.int32)
    output_shape = [batch_size] + list(spatial_shape)
    v2pinds_tensor = scatter_point_inds(indices, point_indices, output_shape)
    return v2pinds_tensor


def sa_create(name, var):
    x = SharedArray.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
