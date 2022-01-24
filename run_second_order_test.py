
from re import X
from simple_slurm import Slurm
import os 
import pandas as pd
import time
'''
    Install Slurm library:
        pip install simple-slurm

    *NOTE:
        Incase of Slurm SBATCH overflow, use kill command below:
        > squeue -u shikhar.srivastava | awk '{print $1}' | xargs -n 1 scancel
        # squeue -n ewc_sensitivity | awk '{print $1}' | xargs -n 1 scancel
'''



def dispatch_job(command, params, N = 1 , n = 1, mem = '100G', \
    cpus_per_task = 128, gres = 'gpu:4', run_name = 'job',\
        output = '/l/users/shikhar.srivastava/workspace/hover_net/logs/slurm/%j.out'):

    '''Dispatch Slurm Job
        Inputs:
        @param command: (str) Python script call command | 
        @param params: (dict) Hypeparameters for the command call | 
    
        Example:
            For command: $python script.py --param1 3.14 --param2 1.618, this will be written as:
            
                command = 'python script.py'
                params['param1'] = 3.14
                params['param2'] = 1.618

                dispatch_job(command, params)
    '''

    print('--- Starting: %s' % run_name)

    slurm = Slurm(N = N, n = n, mem = mem, \
        cpus_per_task = cpus_per_task, gres=gres, time='23:00:00', job_name=run_name,\
            output = output)

    print(slurm)
    for key, value in params.items():
        command += '    --' + str(key) + ' ' + str(value)

    job_id = slurm.sbatch(command, shell ='/bin/bash')
    
    trial_id = '{} > {}'.format(run_name, str(job_id))
    print('Job dispatch details: ', trial_id)
    print(f'command: {command}')



if __name__ == '__main__':

    '''
        ======= DESCRIPTION ======= 
            command: $python script.py --param1 3.14 --param2 1.618
            will be written as:
            command = 'python script.py'
            params['param1'] = 3.14
            params['param2'] = 1.618
            dispatch_job(command, params)
        =============================
    '''
    # Define your command here

    bucket_step_string = 'top19'

    input_dir = '/l/users/shikhar.srivastava/data/pannuke/'
    log_path = '/l/users/shikhar.srivastava/workspace/hover_net/logs/test/second_order/'
    DIR = input_dir + bucket_step_string + '/'

    command = "ulimit -u 10000\n/home/shikhar.srivastava/miniconda3/envs/hovernet_11/bin/python /l/users/shikhar.srivastava/workspace/hover_net/run_test.py"
    params = dict()
    params['gpu'] = '0,1,2,3'
    gres = 'gpu:4'
    params['bucket_step_string'] = bucket_step_string

    selected_types = pd.read_csv(DIR + 'selected_types.csv')['0']

    if not os.path.exists(log_path + bucket_step_string):
        os.makedirs(log_path + bucket_step_string + '/')
        os.makedirs(log_path + bucket_step_string + '/ckpts/')
        os.makedirs(log_path + bucket_step_string + '/slurm/')


    for target_type in selected_types:
        params['target_organ'] = target_type
        for source_type in selected_types:
            params['source_organ'] = source_type
            output = f'{log_path}{bucket_step_string}/slurm/{source_type}-{target_type}-%j.out'
            
            if not os.path.exists(log_path + bucket_step_string + '/ckpts/' + source_type + '-' + target_type + '/'):
                os.makedirs(log_path + bucket_step_string + '/ckpts/' + source_type + '-' + target_type + '/')
            
            if ((os.path.exists(log_path + bucket_step_string + '/ckpts/' + source_type + '-' + target_type + '/net_epoch=2.tar'))\
                & (os.path.exists(log_path + bucket_step_string + '/ckpts/' + source_type + '-' + target_type + '/per_image_stat.pkl'))):
                continue
            else:
                print('--- Starting: %s-%s' % (source_type, target_type))

                dispatch_job(command, params, gres = gres, output=output, run_name = source_type+'-'+target_type+'-'+bucket_step_string)
                time.sleep(50)