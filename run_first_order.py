
from re import X
from simple_slurm import Slurm
import os 
import pandas as pd
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

    bucket_step_string = 'top5'
    run_no = 'wo_imagenet'
    input_dir = '/l/users/shikhar.srivastava/data/pannuke/'
    DIR = input_dir + bucket_step_string + '/'
    command = "ulimit -u 10000\n/home/shikhar.srivastava/miniconda3/envs/hovernet_11/bin/python /l/users/shikhar.srivastava/workspace/hover_net/run_train.py"
    params = dict()
    params['gpu'] = '0,1,2,3'
    params['bucket_step_string'] = bucket_step_string
    params['run_no'] = run_no
    output = f'/l/users/shikhar.srivastava/workspace/hover_net/logs/{run_no}/first_order/{bucket_step_string}/slurm/%j.out'

    selected_types = pd.read_csv(DIR + 'selected_types.csv')['0']

    if not os.path.exists(f'/l/users/shikhar.srivastava/workspace/hover_net/logs/{run_no}/first_order/{bucket_step_string}/'):
        os.makedirs(f'/l/users/shikhar.srivastava/workspace/hover_net/logs/{run_no}/first_order/{bucket_step_string}/')
        os.makedirs(f'/l/users/shikhar.srivastava/workspace/hover_net/logs/{run_no}/first_order/{bucket_step_string}/ckpts/')
        os.makedirs(f'/l/users/shikhar.srivastava/workspace/hover_net/logs/{run_no}/first_order/{bucket_step_string}/slurm/')

    for type in selected_types:
        params['organ'] = type
        if not os.path.exists(f'/l/users/shikhar.srivastava/workspace/hover_net/logs/{run_no}/first_order/{bucket_step_string}/ckpts/{type}/'):
            os.makedirs(f'/l/users/shikhar.srivastava/workspace/hover_net/logs/{run_no}/first_order/{bucket_step_string}/ckpts/{type}/')
        dispatch_job(command, params, output=output, run_name = type+'_'+bucket_step_string)