import argparse
import multiprocessing
import os
import torch

# get number of available gpus
num_gpus = torch.cuda.device_count()
print('num_gpus: {}'.format(num_gpus))

def run_command(data_dir, index):
    command = f"python project_all.py --data_dir={data_dir} --gpus={num_gpus} --index={index} "
    print(command)
    os.system(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--celeb", type=str, help="Name of the celebrity", required=True)
    parser.add_argument('--runs', type=str, default='0-19')
    args = parser.parse_args()
    
    celeb = args.celeb
    
    if '-' in args.runs:
        start, end = args.runs.split('-')
        start, end = int(start), int(end)
        runs = list(range(start, end+1))
    elif ',' in args.runs:
        runs = [int(x) for x in args.runs.split(',')]
    else:
        runs = [int(args.runs)]

    processes = []
    for i in runs:
        data_dir = f"/playpen-nas-ssd/awang/data/eg3d/{celeb}/{i}/train/preprocessed"
        for index in range(num_gpus):
            process = multiprocessing.Process(target=run_command, args=(data_dir, index))
            processes.append(process)
            process.start()
            print('started process!')

        for process in processes:
            process.join()

    print('done!')

