import pandas as pd
import pickle

def read_metric_logs(bucket_type):

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
    return metrics, type_list