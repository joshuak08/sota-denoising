# Denoising InSAR images using State-of-the-Art Image Denoising models

The Restormer architecture in this repository is detailed by [Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang 2022. Restormer: Efficient Transformer for High-Resolution Image Restoration. In CVPR](https://openaccess.thecvf.com/content/CVPR2022/html/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.html). The code presented in this repository is adapted from the [Restormer](https://github.com/swz30/Restormer) repository. 

This model was trained on synthetic InSAR images generated using code adapted from [N. Anantrasirichai, J. Biggs, F. Albino, D. Bull, A deep learning approach to detecting volcano deformation from satellite imagery using synthetic datasets, Remote Sensing of Environment, Volume 230, 2019](https://github.com/pui-nantheera/Synthetic_InSAR_image/tree/main).

## Installation

Follow the steps outlined [INSTALL.md](INSTALL.md) to install the required libraries and dependencies to run this model.

## Training

To train Restormer, go to repository root directory and run:

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt training_options.yml --launcher pytorch
```

Change relevant options in ``training_options.yml`` as required or intended.

## Testing

- To test Restormer on synthetic InSAR images, go to repository root directory and run:

```python
python synthetic_testing.py 
--yaml_file=training_options.yml 
--outputDir='/output' 
--outputImages=True 
--testingImagesDir="path/to/testingImages/dir" # replace with directory to testing images
```

This will produce a spreadsheet to view the evaluation metrics for each saved model state and individual images.

- To test Restormer on real InSAR images, go to repository root directory and run:

```python
python real_testing.py
--yaml_file=training_options.yml  
--outputDir="/output" 
--outputImages=True 
--testingImagesDir="path/to/testingImages/dir" # replace with directory to testing images
--savedState="net_g_latest.pth" # replace with wanted saved model state 
```

Note: the outputDir argument should remain the same for both synthetic and real image testing.

## BlueCrystal4 Usage

If training or testing is done on BlueCrystal4, please following steps instead. Replace the account code in each script as appropriate.

- Training: Go to repository root directory and run

```bash
sbatch bc4_train.sh
```

- Synthetic Testing: Change testing image directory argument in ``bc4_synthetic_testing.sh``. Then run

```bash
sbatch bc4_synthetic_testing.sh
```

- Real Testing: Change testing image directory and saved model state arguments in ``bc4_real_testing.sh``. Then run

```bash
sbatch bc4_real_testing.sh
```

## Convert Tensorboard to graphs

To convert logged information, please run:

```python
python tb_converter.py --path="./tb_logger/name" # replace name relevant model name folder
```

This will produce 3 graphs to show the loss and validation accuracies of different metrics.
