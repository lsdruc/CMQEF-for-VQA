## Context-aware Multi-level Question Embedding Fusion for Visual Question Answering

This repository contains a publicly available code from our paper "Context-aware Multi-level Question Embedding Fusion for Visual Question Answering".

This repository establishes on the re-implementation of publicly available UpDn-based VQA code (https://github.com/hengyuan-hu/bottom-up-attention-vqa). We extend the VQA code, so that it can be used on not only VQAv2 but also VQA-CPv2. 

### Prerequisites

You may need a machine with a GPU with 8GB graphics memory, and PyTorch v-0.3.0 for Python 2.7.

1. Install [PyTorch v-0.3.0](http://pytorch.org/) with CUDA 8.0 and Python 2.7.
2. Install h5py, pillow, and tqdm

### Data Setup

All data should be downloaded to a 'data/' directory in the root
directory of this repository.

The easiest way to download the data is to run the provided script
`tools/download.sh` from the repository root. Then run
`tools/process.sh` from the repository root to process the data to the
correct format.

### Training and Testing

1. Run 'python main.py' to train on the VQAv2 train set, and then run 'python test.py' to test on the VQAv2 validation set.

2. Run 'python main_BDSR.py' to train on the VQAv2 train set and use BDSR to adjust the optimization process, and then run 'python test_BDSR.py' to test on the VQAv2 validation set.

3. Run 'python main_cpv2.py' to train on the VQA-CPv2 train set, and then run 'python test_cpv2.py' to test on the VQA-CPv2 test set.

4. Run 'python main_cpv2_BDSR.py' to train on the VQA-CPv2 train set and use BDSR to adjust the optimization process, and then run 'python test_cpv2_BDSR.py' to test on the VQA-CPv2 test set.

### Results on VQAv2

| Model| Y/N | Num | Other | All |
| --- | --- | --- | --- | --- |
|CMQEF|82.89|47.71|56.64|65.51|
|CMQEF(BDSR)|83.28|48.60|57.06|65.98|

### Results on VQA-CPv2

| Model| Y/N | Num | Other | All |
| --- | --- | --- | --- | --- |
|CMQEF|41.40|12.73|47.81|40.88|
|CMQEF(BDSR)|41.51|12.89|48.01|41.05|

### References

[1] Li S, Gong C, Zhu Y, Luo C, et al. Context-aware Multi-level Question Embedding Fusion for Visual Question Answering. 

[2] Anderson P, He X, Buehler C, et al. Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering [C]. CVPR 2018, Salt Lake City, USA, 2018: 6077–6086.

[3] Teney D, Anderson P, He X, et al. Tips and Tricks for Visual Question Answering: Learnings from the 2017 Challenge [C]. CVPR 2018, Salt Lake City, USA, 2018: 4223-4232.

[4] Hu H. An Efficient PyTorch Implementation of the Winning Entry of the 2017 VQA Challenge [OL]. GitHub, 2017. https://github.com/hengyuan-hu/bottom-up-attention-vqa.
