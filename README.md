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
Execution file, configuration file, and models are download from the [zip](https://drive.google.com/open?id=1vsm1X9isbogaVBLF6R0g7fKgHNjVI14Y&usp=drive_copy) file.  (For reviewers, " ???" is the password to decompress the file.)

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
CUDA_VISIBLE_DEVICES=0,1 python extract_features.py --data_h5_dir DATA_PATCHES --data_slide_dir DATA --csv_path DATA_PATCHES/process_list_autogen.csv --feat_dir DATA_FEATURES --batch_size 128 --slide_ext .ndpi

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
CUDA_VISIBLE_DEVICES=0,1 python extract_features.py --data_h5_dir DATA_PATCHES --data_slide_dir DATA --csv_path DATA_PATCHES/process_list_autogen.csv --feat_dir DATA_FEATURES --batch_size 128 --slide_ext .ndpi --data_augmentation

```
#### 3. Training and Testing List
Prepare the training, validation  and the testing list containing the labels of the files and put it into ./LIST folder. (The csv sample "DATA_train.csv" and  "DATA_test.csv")

example of the csv files:
| slide_id    | case_id  | label |
| :---          | :---           |  :---    |
| slide_1  |slide_1 |  pos      |
| slide_2  | slide_2  | neg      |
|  ...            | ...            | ...        | 
| slide_n  |slide_n   | pos        |   


#### 4. Inference 
For inference, open the "inference.py" and set the number of the classes, the label for each class and the testing list location ("DATA_test.csv").
```
if args.task == 'dummy_mtl_concat':
    args.n_classes=2
    dataset = Generic_MIL_MTL_Dataset(csv_path = 'LIST/DATA_test.csv',
                            data_dir= os.path.join(args.data_root_dir,'pt_files'),
                            shuffle = False, 
                            print_info = True,
                            label_dicts = [{'neg':0, 'pos':1}],
                            label_cols = ['label'],
                            patient_strat= False)
```
To generate the prediction outcome of the MAMILE_CTFE_xxx model, containing K base models:
```
python inference.py  --models_exp_code MAMILE_CTFE_xxx --save_exp_code MAMILE_CTFE_xxx_prediction --results_dir MODELS --data_root_dir DATA_FEATURES --top_fold K 

```
On the other hand, to generate the prediction outcome of the MAMIL_CTFE model, containing one single base models:
```
python inference.py  --models_exp_code MAMILE_CTFE_xxx --save_exp_code MAMIL_CTFE_xxx_prediction --results_dir MODELS --data_root_dir DATA_FEATURES 
```

## Training
#### Preparing Training Splits
To create a splits for training and validation set from the training list automatically. The default proportion for the training: validation splits used in this study is 9:1. Do the stratified sampling by open the create_splits.py, and change this related code with the directory of the training csv, the number of classess and the labels we want to investigates.
```
if args.task == 'dummy_mtl_concat':
    args.n_classes=2
    dataset = Generic_WSI_MTL_Dataset(csv_path = 'LIST/DATA_train.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dicts = [{'neg':0, 'pos':1}],
                            label_cols = ['label'],
                            patient_strat= False)
```
To create a N fold for training and validation set from the training list. The default proportion for the training:validation splits used in this study is 9:1. 
```
SPLIT/
├── splits_0.csv
├── splits_1.csv
│       ⋮
└── splits_N.csv
```
In the terminal run:
```
python create_splits.py --split_dir SPLIT  --k N
```

#### Training
Open the "main.py" and and change this related code with the directory of the training csv, the number of classess and the labels we want to investigates.
```
if args.task == 'dummy_mtl_concat':
    args.n_classes=2
    dataset = Generic_MIL_MTL_Dataset(csv_path ='LIST/DATA_train.csv',
                            data_dir= os.path.join(args.data_root_dir,'pt_files'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dicts = [{'neg':0, 'pos':1}],
                            label_cols = ['label'],
                            patient_strat= False)
```
Run this code in the terminal to train N folds:
```
python main.py --data_root_dir DATA_FEATURES --results_dir MODELS --split_dir SPLIT --exp_code MAMILE_CTFE_xxx --k N

```


## License
This Python source code is released under a creative commons license, which allows for personal and research use only. For a commercial license please contact Prof Ching-Wei Wang. You can view a license summary here:  
http://creativecommons.org/licenses/by-nc/4.0/


## Contact
Prof. Ching-Wei Wang  
  
cweiwang@mail.ntust.edu.tw; cwwang1979@gmail.com  
  
National Taiwan University of Science and Technology
