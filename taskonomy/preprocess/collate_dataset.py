import numpy as np
import os
from utils.data_utils import load_numpy_file, write_numpy_file

def collate_files(folds, input_dir ='/l/users/shikhar.srivastava/data/pannuke/'):
    img = np.empty((0, 256, 256, 3))
    type = np.empty((0))
    mask = np.empty((0, 256, 256, 6))
    for fold_id in range(1, folds+1):
        path = input_dir + 'Fold %d' % fold_id
        img_path = path + '/images/fold%d/images.npy' % fold_id
        type_path = path + '/images/fold%d/types.npy' % fold_id
        mask_path = path + '/masks/fold%d/masks.npy' % fold_id
        img_ = load_numpy_file(img_path)
        types_ = load_numpy_file(type_path)
        masks_ = load_numpy_file(mask_path)
        img = np.concatenate([img,img_], axis = 0)
        type = np.concatenate([type,types_], axis = 0)
        mask = np.concatenate([mask,masks_], axis = 0)
        print(fold_id, "\t", img.shape)
        print(fold_id, "\t", type.shape)
        del img_, types_, masks_
    return img, type, mask

def write_collated(write_dir, folds, **kwargs):

    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    img, type, mask=  collate_files(folds, **kwargs)
    write_numpy_file(write_dir + 'images.npy', img)
    write_numpy_file(write_dir + 'types.npy', type)
    write_numpy_file(write_dir + 'masks.npy', mask)


# Write main

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=3, help='Number of folds')
    parser.add_argument('--write_dir', type=str, default='/l/users/shikhar.srivastava/data/pannuke/full/', help='Directory to write collated files')
    args = parser.parse_args()

    # for the 3 folds, collate and write to dir

    write_collated(args.write_dir, args.folds)

