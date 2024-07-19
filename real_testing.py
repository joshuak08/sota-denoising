import numpy as np
import os
import argparse
import glob
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
import glob
import cv2

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
parser = argparse.ArgumentParser()

parser.add_argument('--yaml_file', required=True)
parser.add_argument('--outputDir', required=True, default="/output")
parser.add_argument('--outputImages', required=True, type=bool)
parser.add_argument('--testingImagesDir', required=True)
parser.add_argument('--savedState', required=True, default='net_g_latest.pth')

args = parser.parse_args()

yaml_file = args.yaml_file
output_dir = args.outputDir
output_images = args.outputImages
testing_dir = args.testingImagesDir
saved_state = args.savedState

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
x['network_g'].pop('type')

name = x['name']

results_dir_png = os.path.join(output_dir + f'/{name}', 'png')
os.makedirs(output_dir, exist_ok=True)

for weight in glob.glob(f'./experiments/{name}/models/{saved_state}'):
    model_no = os.path.split(weight)[-1]
    model_restoration = Restormer(**x['network_g'])
    checkpoint = torch.load(weight)
    model_restoration.load_state_dict(checkpoint['params'])
    print("===>Testing using weights: ", weight)
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()
    with torch.no_grad():
        for file in glob.glob(testing_dir+'/*.png'):
            img = np.float32(cv2.imread(file, cv2.IMREAD_GRAYSCALE))
            img = img[..., np.newaxis]
            img /= 255
            noisy_img = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2)
            restored = model_restoration(noisy_img)
            restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
            
            cleaned_filename = os.path.split(file)[-1]

            if output_images:
                restored_img = img_as_ubyte(restored)
                save_file = os.path.join(results_dir_png, cleaned_filename)
                utils.save_img(save_file, restored_img)
