import torch
from multiprocessing import Pool
import glob
import run_projector
import tempfile
import os
import argparse
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='folder where preprocessed training images are')
parser.add_argument('--index', type=int)
parser.add_argument('--gpus', type=int)
parser.add_argument('--network_pkl', type=str, default='/playpen-nas-ssd/awang/eg3d/eg3d/networks/ffhqrebalanced512-128.pkl')
args = parser.parse_args()

print('running project_all')

num_gpus = args.gpus
out_dir = args.data_dir
network_pkl = args.network_pkl
assert os.path.exists(out_dir), f'folder {out_dir} does not exist'
if out_dir and out_dir[-1] == '/':
    out_dir = out_dir[:-1]

def subprocess_fn(img_path):
    os.chdir('/playpen-nas-ssd/awang/EG3D-projector/eg3d')
    c_path = img_path.replace('png', 'npy')
    if not os.path.exists(os.path.join(out_dir, img_path.replace('.png', '_latent.npy'))):
        command = "python run_projector.py --outdir=" + out_dir + \
            f" --latent_space_type w  --network={network_pkl} --sample_mult=2 " + \
            "  --image_path " + img_path + \
            " --c_path " + c_path
        print('processing: {}'.format(img_path))
        os.system(command)
    else:
        print('skipping: {}'.format(img_path))



pngFilenamesList = []
pngFilenamesList = glob.glob(f'{out_dir}/*.png')
pngFilenamesList = sorted([im for im in pngFilenamesList if not os.path.exists(os.path.join(out_dir, im.replace('.png', '_latent.npy'))) ])
imgs = [pngFilenamesList[i::num_gpus] for i in range(num_gpus)]

# argument for index
index = int(args.index)
os.environ['CUDA_VISIBLE_DEVICES'] = str(index)
imgs = imgs[index]
#imgs = pngFilenamesList
print('imgs: {}'.format(imgs))
import multiprocessing
with multiprocessing.Pool(2) as p: # number of processes per GPU - depends on GPU memory
    p.map(subprocess_fn, imgs)
