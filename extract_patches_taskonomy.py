"""extract_patches.py

Patch extraction script.
"""

import re
import glob
import os
from tqdm import tqdm
import pathlib

import numpy as np

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

from dataset import get_dataset
import numpy as np
import pandas as pd
# -------------------------------------------------------------------------------------

INPUT_DIR = '/l/users/shikhar.srivastava/data/pannuke/top5/'
OUTPUT_DIR = '/l/users/shikhar.srivastava/data/pannuke/processed/top5/'

if __name__ == "__main__":

    # Determines whether to extract type map (only applicable to datasets with class labels).
    type_classification = True

    win_size = [540, 540]
    step_size = [164, 164]
    extract_type = "mirror"  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.

    # Name of dataset - use Kumar, CPM17 or CoNSeP.
    # This used to get the specific dataset img and ann loading scheme from dataset.py
    dataset_name = "pannuke"

    selected_types = pd.read_csv(INPUT_DIR + 'selected_types.csv')['0']

    xtractor = PatchExtractor(win_size, step_size)
    
    splits = ['train', 'test']

    for type in selected_types:

        save_root = OUTPUT_DIR + "%s/" % type

        for split_name in splits:
            
            out_dir = "%s/%s/%dx%d_%dx%d/" % (
            save_root,
            split_name,
            win_size[0],
            win_size[1],
            step_size[0],
            step_size[1],
            )
            rm_n_mkdir(out_dir)
 
            dir = INPUT_DIR + "%s/%s/" % (type, split_name)

        
            # Load the dataset
            imgs = np.load(dir + 'images.npy')
            anns = np.load(dir + 'anns.npy')

            for data_idx, (img, ann) in tqdm(enumerate(zip(imgs, anns)), desc = 'Extracting patches for %s, split[%s]' % (type, split_name)):

                # *
                img = np.concatenate([img, ann], axis=-1)
                sub_patches = xtractor.extract(img, extract_type)
                #print('Extracting patches for image %d/%d' % (data_idx,len(imgs)))
                for idx, patch in enumerate(sub_patches):
                    np.save("{0}/{1}_{2:03d}.npy".format(out_dir, data_idx, idx), patch)