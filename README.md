# Installation
Please follow the instruction of [official SAM 2 repo](https://github.com/facebookresearch/sam2?tab=readme-ov-file#installation). If you encounter issues running the code, it's recommended to create a new environment specifically for SAM2Long instead of sharing it with SAM2. For further details, please check this issue [here](https://github.com/Mark12Ding/SAM2Long/issues/5#issuecomment-2458974462).

```git clone https://github.com/facebookresearch/sam2.git```

make conda environment

```conda env create -f sam2/environment.yml```

Install sam2 (`pip install sam2`) in that environment.

Get SAM2LONG repo

```git clone https://github.com/Mark12Ding/SAM2Long.git```

Install SAM2LONG (`pip install -e .`) in that environment.

## Download Checkpoints
All the model checkpoints can be downloaded by running:
```
bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```
 
# Usage
Once all are installed, use these scripts to run code on MSI.

slurm_script_maker creates a slurm script for running SAM2Long on MSI on the video specified in the command line, with the intended segment object and frame number as arguments.


