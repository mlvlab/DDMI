import pdb
import torch
import numpy as np
import os.path as osp
import argparse
from timeit import default_timer as timer

#from evaluation_metrics import minimum_mathing_distance, jsd_between_point_cloud_sets, coverage
from in_out import snc_category_to_synth_id, load_all_point_clouds_under_folder

from evaluation_metrics_pytorch import compute_all_metrics


def scale_to_unit_sphere(points, center=None):                  # 标准化为半径为1的单位球
    midpoints = (np.max(points, axis=0) + np.min(points, axis=0)) / 2
#    midpoints = np.mean(points, axis=0)
    points = points - midpoints
    scale = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
#    points = points / (scale * 2)
    points = points / scale
    return points

def scale_to_half_unit_sphere(points, center=None):         # 标准化为0.5半径的球
    midpoints = np.mean(points, axis=0)
    points = points - midpoints
    scale = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    points = points / (scale * 2)
    return points


def calculate_mmd_cov(gt_pth, gen_pth):
    all_pc_data = load_all_point_clouds_under_folder(gen_pth, n_threads=8, file_ending='.ply', verbose=True)
    all_pc_data_ref = load_all_point_clouds_under_folder(gt_pth, n_threads=8, file_ending='.ply', verbose=True)

    n_ref = 1355 # size of ref_pcs.
    n_sam = 1355 # size of sample_pcs.
    all_ids = np.arange(all_pc_data.num_examples)
    all_ids_ref = np.arange(all_pc_data_ref.num_examples)
    ref_ids = np.random.choice(all_ids_ref, n_ref, replace=False)
    sam_ids = np.random.choice(all_ids, n_sam, replace=False)
    ref_pcs = all_pc_data_ref.point_clouds[ref_ids]
    sample_pcs = all_pc_data.point_clouds[sam_ids]

    for i in range(n_ref):
        ref_pcs[i] = scale_to_unit_sphere(ref_pcs[i], center=None)
        #ref_pcs[i] = scale_to_half_unit_sphere(ref_pcs[i], center=None)
    for i in range(n_sam):
        sample_pcs[i] = scale_to_unit_sphere(sample_pcs[i], center=None)
        #sample_pcs[i] = scale_to_half_unit_sphere(sample_pcs[i], center=None)

    ae_loss = 'chamfer'  # Which distance to use for the matchings.

    if ae_loss == 'emd':
        use_EMD = True
    else:
        use_EMD = False  # Will use Chamfer instead.
        
    batch_size = 220     # Find appropriate number that fits in GPU.
    normalize = True     # Matched distances are divided by the number of 
                        # points of thepoint-clouds.

    sample_pcs = torch.tensor(sample_pcs).to('cuda')
    ref_pcs = torch.tensor(ref_pcs).to('cuda')

    start = timer()

    result = compute_all_metrics(sample_pcs, ref_pcs, batch_size, accelerated_cd = True)
    print(result)

    end = timer()
    sampling_time = end - start
    print('Time :', sampling_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'DDPM reproduce')
    # Train configuration
    parser.add_argument('--gt_pth', type = str, default = '')
    parser.add_argument('--gen_pth', type = str, default = '')
    args = parser.parse_args()

    calculate_mmd_cov(args.gt_pth, args.gen_pth)


