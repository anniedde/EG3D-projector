import torch
from multiprocessing import Pool
import glob
import run_projector
import tempfile
import os

out_dir = '/playpen-nas-ssd/awang/data/luchao_preprocessed'
   
def subprocess_fn(img_path):
    os.chdir('/playpen-nas-ssd/awang/EG3D-projector/eg3d')
    c_path = img_path.replace('png', 'npy')
    if not os.path.exists(os.path.join(out_dir, img_path.replace('.png', '_latent.npy'))):
        command = "python run_projector.py --outdir=" + out_dir + \
            " --latent_space_type w  --network=networks/ffhq512-128.pkl --sample_mult=2 " + \
            "  --image_path " + img_path + \
            " --c_path " + c_path
        print('processing: {}'.format(img_path))
        os.system(command)

pngFilenamesList = glob.glob('/playpen-nas-ssd/awang/data/luchao_preprocessed/*.png')
pngFilenamesList = [im for im in pngFilenamesList if not os.path.exists(os.path.join(out_dir, im.replace('.png', '_latent.npy'))) ]
#imgs = [pngFilenamesList[i::8] for i in range(8)]
print(pngFilenamesList)
imgs = [pngFilenamesList]
    
# argument for index
import sys, argparse
parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int)
args = parser.parse_args()
index = int(args.index)
# index = 3
os.environ['CUDA_VISIBLE_DEVICES'] = str(index)
imgs = imgs[index]

import multiprocessing
with multiprocessing.Pool(2) as p: # number of processes per GPU - depends on GPU memory
    p.map(subprocess_fn, imgs)

# with multiprocessing.Pool(5) as p:
#     p.map(helper, imgs)


"""
def subprocess_fn(img_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Init torch.distributed.
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=0, world_size=8)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=0, world_size=8)

    # Init torch_utils.
    sync_device = torch.device('cuda') if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=0, sync_device=sync_device)

    params_path = img_path.replace('png', 'npy')

    run_projector.run_direct(
        network_pkl='networks/ffhq512-128.pkl',
        outdir=out_dir,
        sampling_multiplier=2,
        latent_space_type='w',
        image_path=img_path,
        c_path=params_path,
        num_steps=500,
        nrr=None
    )   
"""