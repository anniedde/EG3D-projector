import json
import numpy
import os
import dnnlib
import torch
import wget
import run_projector
import glob

"""
out_dir = '/playpen-nas-ssd/awang/data/luchao_preprocessed'

pngFilenamesList = glob.glob('/playpen-nas-ssd/awang/data/luchao_preprocessed/*.png')
count = 0
for im in pngFilenamesList:
    if not os.path.exists(os.path.join(out_dir, im.replace('.png', '_latent.npy'))):
        print(im)
        count += 1
print(count) 

"""
run_projector.run_direct(
        network_pkl='/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/utils/trained_luchao_50_images_no_lora.pkl',
        outdir=out_dir,
        sampling_multiplier=2,
        latent_space_type='w',
        image_path='',
        c_path='',
        num_steps=500,
        nrr=None
    ) 
"""
for num in range(1, 1000):
    numString = (str)(num).rjust(8, '0') 
    imgname = '/playpen-nas-ssd/awang/data/ffhq_subset/00000/' + numString + '.png'
    camname = '/playpen-nas-ssd/awang/data/ffhq_subset/00000/' + numString + '.npy'
    run_projector.run_direct(
        network_pkl='networks/ffhq512-128.pkl',
        outdir='/playpen-nas-ssd/awang/data/ffhq_subset/00000/',
        sampling_multiplier=2,
        latent_space_type='w',
        image_path=imgname,
        c_path=camname,
        num_steps=500,
        nrr=None
    )   


run(
        network_pkl='networks/ffhq512-128.pkl',
        outdir='projector_out',
        sampling_multiplier=2,
        latent_space_type='w',
        image_path:str,
        c_path:str,
        num_steps=500
)
"""


url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
#output_directory = '/playpen-nas-ssd/awang/EG3D-projector/eg3d/network'

#filename = wget.download(url, out=output_directory)

# Load VGG16 feature detector.
url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
#url = './networks/vgg16.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#with dnnlib.util.open_url(url) as f:
#    vgg16 = torch.jit.load(f).eval().to(device)

#with dnnlib.util.open_url(url) as f:
#    device = 'cuda' if torch.cuda.is_available() else 'cpu'
#    vgg16 = torch.jit.load(f).eval().to(device)
#    torch.save(vgg16, '/playpen-nas-ssd/awang/EG3D-projector/eg3d/network/vgg16.pt')
