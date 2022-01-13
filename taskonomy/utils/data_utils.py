import numpy as np
from scipy.optimize import linear_sum_assignment

import numpy as np
# Load images from numpy array
def load_numpy_file(path):
    # Load numpy array
    arr = np.load(path)
    # Return a list of numpy arrays
    return arr
# Write numpy files to path
def write_numpy_file(path, arr):
    np.save(path, arr)
    return

def randint(max_val, num_samples):
    """
    return num_samples random integers in the range(max_val)
    """
    rand_vals = {}
    _num_samples = min(max_val, num_samples)
    while True:
        _rand_vals = np.random.randint(0, max_val, num_samples)
        for r in _rand_vals:
            rand_vals[r] = r
            if len(rand_vals) >= _num_samples:
                break

        if len(rand_vals) >= _num_samples:
            break
    return rand_vals.keys()
    
####
def get_fast_pq(true, pred, match_iou=0.5):
    """
    `match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique 
    (1 prediction instance to 1 GT instance mapping).

    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing. 

    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.    
    
    Fast computation requires instance IDs are in contiguous orderding 
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand 
    and `by_size` flag has no effect on the result.

    Returns:
        [dq, sq, pq]: measurement statistic

        [paired_true, paired_pred, unpaired_true, unpaired_pred]: 
                      pairing information to perform measurement
                    
    """
    assert match_iou >= 0.0, "Cant' be negative"
    
    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [None,]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)
    
    pred_masks = [None,]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_iou = np.zeros([len(true_id_list) -1, 
                             len(pred_id_list) -1], dtype=np.float64)

    # caching pairwise iou
    for true_id in true_id_list[1:]: # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0: # ignore
                continue # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id-1, pred_id-1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1 # index is instance id - 1
        paired_pred += 1 # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence 
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum   
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair 
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]
#####

def remap_label(pred, by_size=False):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1  
    return new_pred
####


def binarize(x):
    '''
    convert multichannel (multiclass) instance segmetation tensor
    to binary instance segmentation (bg and nuclei),

    :param x: B*B*C (for PanNuke 256*256*5 )
    :return: Instance segmentation
    '''
    out = np.zeros([x.shape[0], x.shape[1]])
    count = 1
    for i in range(x.shape[2]):
        x_ch = x[:,:,i]
        unique_vals = np.unique(x_ch)
        unique_vals = unique_vals.tolist()
        unique_vals.remove(0)
        for j in unique_vals:
            x_tmp = x_ch == j
            x_tmp_c = 1- x_tmp
            out *= x_tmp_c
            out += count*x_tmp
            count += 1
    out = out.astype('int32')
    return out
####

def get_tissue_idx(tissue_indices, idx):
    for i in range(len(tissue_indices)):
        if tissue_indices[i].count(idx) == 1:
            tiss_idx = i
    return tiss_idx 


def binarize_type(x):
    '''
    convert multichannel (multiclass) instance segmetation tensor
    to single channel type segmentation (bg and nuclei types),

    :param x: B*B*C (for PanNuke 256*256*5 )
    :return: Type segmentation [256, 256] where each pixel is [0,C] with 0 as background and C classes
    '''
    out = np.zeros([x.shape[0], x.shape[1]])
    #count = 1
    for i in range(x.shape[2]):
        x_ch = x[:,:,i]
        x_tmp = x_ch > 0
        x_tmp_c = 1 - x_tmp
        out *= x_tmp_c
        out += (i+1)*x_tmp

    out = out.astype('int32')
    return out

from typing import List
from tqdm import tqdm
def pre_processing_mask(masks: np.array) -> List[np.array]: 
    ''' Converts Mask with shape [N, 256, 256, 6] with Channels (0 to 4 of Classes, 5: Background)  
        to:
            ann_type: [N, 256, 256] with per image-processing to get each pixel value equal to [1,5] and Background as 0
            ann_inst: [N, 256, 256] with per image-processing to get each pixel value equal to [0,N] with Background as 0 and N as counts of\
                      no of Nuclei instances per image

    '''
    ann_type = np.zeros([masks.shape[0], masks.shape[1], masks.shape[2]], dtype=np.int32)
    ann_inst = np.zeros([masks.shape[0], masks.shape[1], masks.shape[2]], dtype=np.int32)
    # Looping over images
    for i in tqdm(range(masks.shape[0]),desc='Generating Instance, Type Masks'):
        x_inst = binarize(masks[i,:,:,:5])
        x_type = binarize_type(masks[i,:,:,:5])
        ann_type[i,:,:] = x_type
        ann_inst[i,:,:] = x_inst

    return ann_type, ann_inst