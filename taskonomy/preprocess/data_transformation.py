import os
from cv2 import threshold
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.data_utils import *
import matplotlib.pyplot as plt

def bucketing(write_dir, input_dir ='/l/users/shikhar.srivastava/data/pannuke/full/', \
    training_splits = ['train', 'test'], training_split_ratios = [0.8,0.2], bucketing_threshold = 224, split_viz_path = '/l/users/shikhar.srivastava/workspace/hover_net/taskonomy/viz/data_split_viz/'):
    """
    Bucketing the data by Organ Location, within train and test splits.
    """
    assert np.sum(training_split_ratios) == 1.0, "Sum of training split ratios should be 1"
    assert len(training_splits) == len(training_split_ratios), "Training splits and training split ratios should be same shape"

    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    if not os.path.exists(split_viz_path):
        os.makedirs(split_viz_path)

    img = load_numpy_file(input_dir + 'images.npy')
    types = load_numpy_file(input_dir + 'types.npy')
    masks = load_numpy_file(input_dir + 'masks.npy')

    type_mask, inst_mask = pre_processing_mask(masks)
    ann = np.stack([inst_mask, type_mask], axis=-1)
    ann = ann.astype("int32")

    type_counts = pd.DataFrame(types).value_counts()
    selected_types = type_counts[type_counts >= bucketing_threshold]
    # Log selected_types
    selected_types.to_csv(write_dir + 'selected_types.csv')
    

    for selected_type in tqdm(selected_types.index, desc = 'Iterating Buckets'):
        type_name = selected_type[0]
        type_indices = np.where(types == type_name)[0]
        rand_select = list(randint(len(type_indices), bucketing_threshold))
    
        indices_of_given_type = list(np.array(type_indices)[rand_select])
        #print(np.array(indices_of_given_type).shape)

        indices_taken = 0
        for split_idx, split in enumerate(training_splits):

            if not os.path.exists(write_dir + type_name):
                os.makedirs(write_dir + type_name)
            if not os.path.exists(write_dir + type_name + '/' + split):
                os.makedirs(write_dir + type_name + '/' + split)

            thresh = int(bucketing_threshold * training_split_ratios[split_idx]) # No of images per split
            # Randomly select indices for this split
            
            thresh_indices = indices_of_given_type[indices_taken:indices_taken+thresh]
            indices_taken += thresh
            # Write selected files
            write_numpy_file(write_dir + type_name + '/' + split + '/images.npy', img[thresh_indices])
            write_numpy_file(write_dir + type_name + '/' + split + '/types.npy', types[thresh_indices])
            write_numpy_file(write_dir + type_name + '/' + split + '/anns.npy', ann[thresh_indices])

            # Save Split Viz to viz folder
            _types = type_mask[thresh_indices]
            _types = _types[_types!=0]
            pd.DataFrame(_types.flatten()).value_counts().plot(kind='pie', title = '%s [%s] Class-wise Distribution'%(type_name, split), legend = True,figsize=(8,8)).get_figure().savefig(split_viz_path + '%s_%s_pie.png'%(type_name, split))
            plt.clf()

    print('\n')
    print('Bucketing Done')
    print('\n Type counts are: ',type_counts)
    print('\n Selected Types are: \n', selected_types)


# Write main

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--write_dir', type=str, default='/l/users/shikhar.srivastava/data/pannuke/proc10/', help='Directory to write bucketed, processed data')
    parser.add_argument('--input_dir', type=str, default='/l/users/shikhar.srivastava/data/pannuke/full/', help='Directory to read original dataset')
    parser.add_argument('--training_splits', type=list, default='train,test', help='List of train/test/valid splits')
    parser.add_argument('--training_split_ratios', type=list, default='0.8,0.2', help='List of train/test/valid split ratios')
    parser.add_argument('--bucketing_threshold', type=int, default=224, help='Threshold for bucketing')

    args = parser.parse_args()

    bucketing(args.write_dir, args.input_dir, args.training_splits, args.training_split_ratios, args.bucketing_threshold)


