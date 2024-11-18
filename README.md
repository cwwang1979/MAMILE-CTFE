# MAMILE-CTFE
## Setup

#### Requirerements
- Linux (Tested on Ubuntu 18.04)
- NVIDIA GPU (Tested on a single NVIDIA GeForce GTX 1080 Ti)
- RAM >= 16 GB
- GPU Memory >= 12 GB
- GPU driver version >= 470.223.02 
- CUDA version >= 11.4
- Python (3.7.16), h5py (3.8.0), matplotlib (3.5.3), numpy (1.21.6), opencv-python (4.8.1.78), openslide-python (1.2.0), pandas (1.3.5), pillow (9.5.0), PyTorch (1.13.1+cu117), scikit-learn (1.0.2), scipy (1.7.3), tensorflow (1.14.0), tensorboardx (2.6.2.2), torchvision (0.14.1), pixman(0.38.0), huggingface-hub(0.16.4).

#### Download
Execution file, configuration file, and models are download from the [zip](???) file.  (For reviewers, " ???" is the password to decompress the file.)

## Steps
#### 1.Installation

Please refer to the following instructions.
```
# create and activate the conda environment
conda create -n tmil python=3.7 -y
conda activate tmil

# install pytorch
## pip install
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
## conda install
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

# install related package
pip install -r requirements.txt
```

#### 1. Tissue Segmentation and Patching

Place the whole slide image in ./DATA
```
./DATA/
├── slide_1.ndpi
├── slide_2.ndpi
│        ⋮
└── slide_n.ndpi
  
```

For cytology smear dataset, run in terminal:
```
python create_patches.py --source DATA --save_dir DATA_PATCHES --patch_size 224 --preset cyto.csv --seg --patch --stitch

```
For cell block dataset, run in terminal:
```
python create_patches.py --source DATA --save_dir DATA_PATCHES --patch_size 224 --preset cellblock.csv --seg --patch --stitch

```

The result will be produced in folder named 'DATA_PATCHES/', which includes the masks and the sticthes in .jpg and the coordinates of the patches will stored into HD5F files (.h5) like the following structure.
```
DATA_PATCHES/
├── masks/
│   ├── slide_1.jpg
│   ├── slide_2.jpg
│   │       ⋮
│   └── slide_n.jpg
│
├── patches/
│   ├── slide_1.h5
│   ├── slide_2.h5
│   │       ⋮
│   └── slide_n.h5
│
├── stitches/
│   ├── slide_1.jpg
│   ├── slide_2.jpg
│   │       ⋮
│   └── slide_n.jpg
│
└── process_list_autogen.csv
```

#### 2. Feature Extraction

Request access to the UNI model weights from the Huggingface model page at: <https://huggingface.co/mahmoodlab/UNI>

In the terminal run:
```
CUDA_VISIBLE_DEVICES=0,1 python extract_features.py --data_h5_dir DATA_PATCHES --data_slide_dir DATA --csv_path DATA_PATCHES/process_list_autogen.csv --feat_dir DATA_FEATURES --batch_size 512 --slide_ext .ndpi

```
example features results:
```
DATA_FEATURES/
├── h5_files/
│   ├── slide_1.h5
│   ├── slide_2.h5
│   │       ⋮
│   └── slide_n.h5
│
└── pt_files/
    ├── slide_1.pt
    ├── slide_2.pt
    │       ⋮
    └── slide_n.pt
```
Add data augmentation:
```
CUDA_VISIBLE_DEVICES=0,1 python extract_features.py --data_h5_dir DATA_PATCHES --data_slide_dir DATA --csv_path DATA_PATCHES/process_list_autogen.csv --feat_dir DATA_FEATURES --batch_size 512 --slide_ext .ndpi --data_augmentation

```
#### 3. Training and Testing List
Prepare the training, validation  and the testing list containing the labels of the files and put it into ./dataset_csv folder. (The csv sample "fold0.csv" is provided)

example of the csv files:
|      | train          | train_label     | val        | val_label | test        | test_label |  
| :--- | :---           |  :---           | :---:      |:---:      | :---:      |:---:      | 
|  0   | train_slide_1        | 1               | val_slide_1    |   0       | test_slide_1    |   0       | 
|  1   | train_slide_2        | 0               | val_slide_2    |   1       | test_slide_2    |   0       |
|  ... | ...            | ...             | ...        | ...       | ...        | ...       |
|  n-1   | train_slide_n        | 1               |     |          |    |          |



#### 4. Inference 

To generate the prediction outcome of the ETMIL model, containing K base models:
```
python ensemble_inf.py --stage='test' --config='Config/TMIL.yaml'  --gpus=0 --top_fold=K
```
On the other hand, to generate the prediction outcome of the TMIL model, containing one single base models:
```
python ensemble_inf.py --stage='test' --config='Config/TMIL.yaml'  --gpus=0 --top_fold=1
```

To setup the ETMIL model for diffierent tasks: 
1. Open the Config file ./Config/TMIL.yaml
2. Change the log_path in Config/TMIL.yaml to the correlated model path
   
(e.g. For prediction of the cancer subtype in CRC: please set the parameter "log_path" in Config/TMIL.yaml as "./log/TCGA_CRC/CRC_subtype/ETMIL_SSLViT/")

The model of each task has been stored in the zip file with the following file structure: 
```
log/
├── TCGA_CRC/
│   ├── CRC_subtype
│   │   └── ETMIL_SSLViT
│   │
│   ├── Mucinous_TMB_status 
│   │   └── ETMIL_SSLViT
│   │
│   └── Non-mucinous_TMB_status
│       └── ETMIL_SSLViT
│
└── TCGA_EC/
    ├── EC_subtype
    │   └── ETMIL_SSLViT
    │
    ├── Aggressive_TMB_status
    │   └── ETMIL_SSLViT
    │
    └── Non-aggressive_TMB_status
        └── ETMIL_SSLViT      
```


## Training
#### Preparing Training Splits

To create a N fold for training and validation set from the training list. The default proportion for the training:validation splits used in this study is 9:1. 
```
dataset_csv/
├── fold0.csv
├── fold1.csv
│       ⋮
└── foldN.csv
```

#### Training

Run this code in the terminal to training N fold:
```
for((FOLD=0;FOLD<N;FOLD++)); do python train.py --stage='train' --config='Config/TMIL.yaml' --gpus=0 --fold $FOLD ; done
```

Run this code in the terminal to training one single fold:
```
python train.py --stage='train' --config='Config/TMIL.yaml' --gpus=0 --fold=0
```


## License
This Python source code is released under a creative commons license, which allows for personal and research use only. For a commercial license please contact Prof Ching-Wei Wang. You can view a license summary here:  
http://creativecommons.org/licenses/by-nc/4.0/


## Contact
Prof. Ching-Wei Wang  
  
cweiwang@mail.ntust.edu.tw; cwwang1979@gmail.com  
  
National Taiwan University of Science and Technology
