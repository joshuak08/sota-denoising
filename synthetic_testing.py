import numpy as np
import os
import argparse

import glob
import xlsxwriter
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

args = parser.parse_args()

yaml_file = args.yaml_file
output_dir = args.outputDir
output_images = args.outputImages
testing_dir = args.testingImagesDir

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
x['network_g'].pop('type')

name = x['name']

results_dir_png = os.path.join(output_dir + f'/{name}', 'png')
os.makedirs(output_dir, exist_ok=True)

excel_filename = output_dir + '/results.xlsx'
workbook = xlsxwriter.Workbook(excel_filename)

# directory containing test images
noisy_test_images_src = './Volcano/Datasets/new_test/DST/'
clean_test_images_src = './Volcano/Datasets/new_test/D/'
save_images = False

weights = f'./experiments/{name}/models/net_g_220000.pth'

for weight in glob.glob(weights):
    model_no = os.path.split(weight)[-1]
    worksheet = workbook.add_worksheet(model_no)
    row = 0
    worksheet.write(row, 0, 'image_no')
    worksheet.write(row, 1, 'psnr')
    worksheet.write(row, 2, 'ssim')
    worksheet.write(row, 3, 'mse')
    row += 1

    result = {'psnr': [], 'ssim': [], 'mse': []}
    model_restoration = Restormer(**x['network_g'])
    checkpoint = torch.load(weight)
    model_restoration.load_state_dict(checkpoint['params'])
    print("===>Testing using weights: ", weight)
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()
    with torch.no_grad():
        for file in glob.glob(noisy_test_images_src+'*.png'):
            img = np.float32(cv2.imread(file, cv2.IMREAD_GRAYSCALE))
            img /= 255
            img = img[...,np.newaxis]
            noisy_img = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2)
            restored = model_restoration(noisy_img)
            restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
            
            cleaned_filename = os.path.split(file)[-1]
            restored_img = img_as_ubyte(restored)

            if output_images:
                save_file = os.path.join(results_dir_png, cleaned_filename)
                utils.save_img(save_file, restored_img)
            
            restored_img = np.float32(restored_img)
            gt_file = cv2.imread(clean_test_images_src + cleaned_filename, cv2.IMREAD_GRAYSCALE)
            gt_file = np.float32(gt_file)
            gt_file = gt_file[...,np.newaxis]

            psnr = utils.calculate_psnr(gt_file, restored_img)
            ssim = utils.calculate_ssim(gt_file, restored_img)
            mse = utils.calculate_mse(gt_file, restored_img)

            result['psnr'].append(psnr)
            result['ssim'].append(ssim)
            result['mse'].append(mse)

            worksheet.write(row, 0, cleaned_filename)
            worksheet.write(row, 1, psnr)
            worksheet.write(row, 2, ssim)
            worksheet.write(row, 3, mse)
            row += 1

    worksheet.write(2, 6, 'Avg PSNR')
    worksheet.write(2, 7, np.mean(result['psnr']))
    worksheet.write(3, 6, 'Min PSNR')
    worksheet.write(3, 7, np.min(result['psnr']))
    worksheet.write(4, 6, 'Max PSNR')
    worksheet.write(4, 7, np.max(result['psnr']))

    worksheet.write(6, 6, 'Avg SSIM')
    worksheet.write(6, 7, np.mean(result['ssim']))
    worksheet.write(7, 6, 'Min SSIM')
    worksheet.write(7, 7, np.min(result['ssim']))
    worksheet.write(8, 6, 'Max SSIM')
    worksheet.write(8, 7, np.max(result['ssim']))

    worksheet.write(10, 6, 'Avg MSE')
    worksheet.write(10, 7, np.mean(result['mse']))
    worksheet.write(11, 6, 'Min MSE')
    worksheet.write(11, 7, np.min(result['mse']))
    worksheet.write(12, 6, 'Max MSE')
    worksheet.write(12, 7, np.max(result['mse']))
workbook.close()

