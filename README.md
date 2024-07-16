# [ECCV 2024] CoSIGN: Few-Step Guidance of ConSIstency Model to Solve General INverse Problems

This is the official codebase for CoSIGN: Few-Step Guidance of ConSIstency Model to Solve General INverse Problems. We built our repository based on  [openai/consistency-models](https://github.com/openai/consistency_models).

# Pre-trained models

We conducted our experiments on two datasets: LSUN Bedroom-256 and AAPM LDCT.

For natural image restoration, we train the ControlNet over the consistency model checkpoint pre-trained on LSUN bedroom. The pre-trained checkpoint provided by OpenAI can be downloaded [here](https://openaipublic.blob.core.windows.net/consistency/cd_bedroom256_lpips.pt). For medical image restoration, we trained the diffusion model and ditilled the consistency model from it on our own. The checkpoint of the resulting consistency model can be downloaded [here](https://drive.google.com/file/d/1j17aTgmfyEBqGvkgS1Yy7VAdYK8uLXDg/view?usp=sharing). Please create a directory named "checkpoint/" and put all the checkpoints under it.

Here are the download links to checkpoints of the ControlNet in the final CoSIGN model:

### Natural Image: LSUN Bedroom-256

 * Central block inpainting: [inpaint.pt](https://drive.google.com/file/d/1pgOzYEeFrzsx2RdP0uLJ8Vw7Sn1_Wjaf/view?usp=sharing)
 * 4x Super resolution: [super_resolution.pt](https://drive.google.com/file/d/1MI0F62YTfdD77d07OJoAslSY1LMiPN_U/view?usp=sharing)
 * Nonlinear deblur: [nonlinear.pt](https://drive.google.com/file/d/1aO4-RnqasbP5Y96dhRt3AgIirElAq8b5/view?usp=sharing)

### Medical Image: AAPM LDCT

 * Sparse-view CT reconstruction: [ct_recon.pt](https://drive.google.com/file/d/1rnTcR_b-S1QOWaeJ7SX60BQwvEr4fg45/view?usp=sharing)

# Dependencies

To install all packages in this codebase along with their dependencies, run
```bash
source scripts/setup.sh
```

# Downloading datasets

This directory includes instructions and scripts for downloading LSUN bedrooms for use in this codebase. Due to privacy issues, we did not provide instructions on AAPM LDCT dataset.

### LSUN bedroom

Before going through this part, please change your directory to the `datasets` directory:
```bash
cd datasets
```

To download and pre-process LSUN bedroom, clone [fyu/lsun](https://github.com/fyu/lsun) on GitHub and run their download script `python3 download.py -c bedroom`. The result will be an "lmdb" database named like `bedroom_train_lmdb`.

For validation, you need to convert the `bedroom_val_lmdb` database into an image folder. You can pass this to our [lsun_bedroom.py](lsun_bedroom.py) script like so:

```bash
python lsun_bedroom.py bedroom_val_lmdb lsun_val
```

### AAPM LDCT

For training, please pack the CT slices into a .npy path with shape `[num_imgs, 256, 256]`. Name it ldct_train.npy and place it under the `datasets/` path. For validation, please construct an imagefolder called "ldct_val", in which images are named as "test_xxx.png".

Finally, your `datasets` directory should look like this:
```bash
datasets
├── bedroom_train_lmdb
├── ldct_train.npy
├── lsun_val
│   ├── bedroom_0000000.png
│   ├── ···
│   └── bedroom_0000300.png
├── ldct_val
│   ├── test_000.png
│   ├── ···
│   └── test_300.png
└── lsun_bedroom.py
```

# Model training and sampling

Taking inpainting as an example, We provide commands for model training and sampling below. Please change the parameter -t for into super_resolution, nonlinear_deblur or ct_recon for other tasks.

Before running these command, please fill in necessary paths and parameters in the "Build Experiment Environment" part in launch.sh.

```bash
# cosign training
bash scripts/launch.sh -t inpainting -s train_cc
# cosign single-step sampling
bash scripts/launch.sh -t inpainting -s sample_cc --control_net_ckpt /path/to/ckpt
# cosign multi-step sampling
bash scripts/launch.sh -t inpainting -s sample_cc --ts 0,17,39 --control_net_ckpt /path/to/ckpt [--hard_constraint]

# [below for CT recon only]
# edm training
bash scripts/launch.sh -t ct_recon -s train_edm
# edm sampling 
bash scripts/launch.sh -t ct_recon -s sample_edm

# cm training
bash scripts/launch.sh -t ct_recon -s train_cm
# cm sampling
bash scripts/launch.sh -t ct_recon -s sample_cm
```

If you are using distributed training and sampling, please modify `GPUS_PER_NODE` at line 16 of cc/dist_util according to your cluster layout. 

# Evaluations

To compare different generative models, we use FID, Precision, Recall, and Inception Score. These metrics can all be calculated using batches of samples stored in `.npz` (numpy) files. One can evaluate samples with [cm/evaluations/evaluator.py](evaluations/evaluator.py) in the same way as described in [openai/guided-diffusion](https://github.com/openai/guided-diffusion), with reference dataset batches provided therein.

# Citation

Please consider citing this paper if you find the code useful

```bibtex
@article{cosign,
  title={CoSIGN: Few-Step Guidance of ConSIstency Model to Solve General INverse Problems},
  author={Zhao, Jiankun and Song, Bowen and Shen, Liyue},
  journal={European Conference on Computer Vision},
  year={2024}
}
```
