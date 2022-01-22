import pandas as pd
import numpy as np
import pickle

def compute_winrate_ij(source_i, source_j, target_type, metrics):
    metrics_source_i = metrics[(metrics['source_type'] == source_i) & (metrics['target_type'] == target_type)]['stats'].iloc[0]
    metrics_source_j = metrics[(metrics['source_type'] == source_j) & (metrics['target_type'] == target_type)]['stats'].iloc[0]
    n_images = len(metrics_source_i)
    wins = 0
    losses = 0
    for img_idx in range(n_images):
        acc_i = metrics_source_i[img_idx]['np_acc']
        acc_j = metrics_source_j[img_idx]['np_acc']
        dice_i = metrics_source_i[img_idx]['np_dic']
        dice_j = metrics_source_j[img_idx]['np_dic']
        if (dice_i > dice_j):
            wins += 1
        else:
            losses += 1
    winrate = wins/n_images
    losses = losses/n_images
    return winrate #, losses

def tournament_matrix(type_list, metrics):
    W = np.zeros(shape = (len(type_list), len(type_list), len(type_list)))
    labels = []
    for t, target_type in enumerate(type_list):
        sources = type_list
        #sources.remove(target_type)
        print('target_type: ', target_type)
        n_sources = len(sources) # All but the target, can be source types
        W_t = np.zeros((n_sources, n_sources))
        labels_t = list(sources)

        for i, source_i in enumerate(sources):
            for j, source_j in enumerate(sources):
                #print(source_i, source_j)
                W_t[i, j] = compute_winrate_ij(source_i, source_j, target_type, metrics)
                #acc_pivot.loc[source_i, target_type] - acc_pivot.loc[source_j, target_type]
        W_t = np.clip(W_t, 1e-3, 1 - 1e-3)
        W_t_ = W_t.copy()
        W_t_ = (W_t_/W_t_.T)
        assert np.array_equal(W_t_.astype(np.float32), (1/W_t_.T).astype(np.float32)) == True , 'Positive Reciprocal Matrix Assertion Error.' 
        
        labels.append(labels_t)
        W[t, :, :] = W_t_

    return W.astype(np.float32), labels

def priority_vectors(W):
    n_buckets = W.shape[0]
    u,v = np.linalg.eig(W)
    p_values = u[:,0].astype(np.float32)
    p_vectors = v[:,:,0].astype(np.float32)
    p_vectors = np.array([p_vectors[i,:] / p_vectors[i,:].sum() for i in range(p_vectors.shape[0])])
    return p_vectors, p_values

def affinity_distance(p_vectors, beta = 20):
    return np.exp(-beta * p_vectors)

def calculate_affinity_distance(metrics, type_list, beta = 20):
    W, labels = tournament_matrix(type_list, metrics)
    p_vectors, p_values = priority_vectors(W)
    D = affinity_distance(p_vectors, beta)
    return D, labels

# Write main
if __name__ == '__main__':

    bucket_type = 'top5'

    metrics = pd.DataFrame(columns=['source_type', 'target_type', 'stats'])

    type_list_path = f'/l/users/shikhar.srivastava/data/pannuke/{bucket_type}/selected_types.csv'
    type_list = pd.read_csv(type_list_path)['0']

    for source_type in type_list:
        for target_type in type_list:
            logs_path = f'/l/users/shikhar.srivastava/workspace/hover_net/logs/test/second_order/{bucket_type}/ckpts/{source_type}-{target_type}/per_image_stat.pkl'
            # Read pickle file
            with open(logs_path, 'rb') as f:
                stats = pickle.load(f)
            
            metrics = metrics.append({'source_type': source_type, 'target_type': target_type, 'stats': stats}, ignore_index=True)
    
    W, labels = tournament_matrix(type_list, metrics)
    p_vectors, p_values = priority_vectors(W)
    beta = 15
    a_distances = affinity_distance(p_vectors, beta=beta)
    
    print(a_distances)