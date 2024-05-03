import pdb
import trimesh
import argparse
import os
import os.path as osp
import re
import numpy as np
from multiprocessing import Pool
from pc_utils import write_ply

def files_in_subdirs(top_dir, search_pattern):
    regex = re.compile(search_pattern)
    for path, _, files in os.walk(top_dir):
        for name in files:
            full_name = osp.join(path, name)
            if regex.search(full_name):
                yield full_name

def load_point_clouds_from_filenames(file_names, n_threads, loader, verbose=False):
    pc = loader(file_names[0])[0]
    pclouds = np.empty([len(file_names), pc.shape[0], pc.shape[1]], dtype=np.float32)
    model_names = np.empty([len(file_names)], dtype=object)
    class_ids = np.empty([len(file_names)], dtype=object)
    pool = Pool(n_threads)

    for i, data in enumerate(pool.imap(loader, file_names)):
        pclouds[i, :, :], model_names[i], class_ids[i] = data

    pool.close()
    pool.join()

    if len(np.unique(model_names)) != len(pclouds):
        warnings.warn('Point clouds with the same model name were loaded.')

    if verbose:
        print('{0} pclouds were loaded. They belong in {1} shape-classes.'.format(len(pclouds), len(np.unique(class_ids))))

    return pclouds, model_names, 

def pc_loader(f_name):
    ''' loads a point-cloud saved under ShapeNet's "standar" folder scheme: 
    i.e. /syn_id/model_name.ply
    '''
    tokens = f_name.split('/')
    model_id = tokens[-1].split('.')[0]
    synet_id = tokens[-2]
    return load_ply(f_name), model_id, synet_id

def load_ply(file_name, with_faces=False, with_color=False):
    #ply_data = PlyData.read(file_name)
    ply_data = trimesh.exchange.off.load_off(file_name)
    points = ply_data['vertex']
    points = np.vstack([points['x'], points['y'], points['z']]).T
    ret_val = [points]

    if with_faces:
        faces = np.vstack(ply_data['face']['vertex_indices'])
        ret_val.append(faces)

    if with_color:
        r = np.vstack(ply_data['vertex']['red'])
        g = np.vstack(ply_data['vertex']['green'])
        b = np.vstack(ply_data['vertex']['blue'])
        color = np.hstack((r, g, b))
        ret_val.append(color)

    if len(ret_val) == 1:  # Unwrap the list
        ret_val = ret_val[0]

    return ret_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'DDPM reproduce')
    # Train configuration
    parser.add_argument('--pth', type = str, default = '')
    parser.add_argument('--save_pth', type = str, default = '')
    args = parser.parse_args()

    file_names = [f for f in files_in_subdirs(args.pth, '.obj')]
    for idx, i in enumerate(file_names):
        print(i)
        recon_mesh = trimesh.load(i)
        recon_pts, _ = trimesh.sample.sample_surface(recon_mesh, 2048)
        save_path = args.save_pth + str(idx)+'.ply'
        write_ply(recon_pts, save_path)
    
