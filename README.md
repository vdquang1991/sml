# SELECTIVE MUTUAL LEARNING: AN EFFICIENT APPROACH FOR SINGLE CHANNEL SPEECH SEPARATION

By Ha Minh Tan, Duc-Quang Vu, Chung-Ting Lee, Yung-Hui Li, Jia-Ching Wang,
## Overview
We implement SML for the single-channel speech separation model in the time domain. 
Our SML approach that only high-confidence predictions are utilized to guide the remaining 
network while the low-confidence predictions are ignored. 

The figure below shows SML approach.
<p align="center">
  <img width="800" alt="fig_method" src="https://github.com/vdquang1991/selective_mutual_learning/blob/main/sml.png">
</p>


## Running the code

### Requirements
- Python3
- Tensorflow (>=2.3.0)
- numpy 
- Pillow
- librosa
- pystoi
- pesq
- mir_eval

### Training

In this code, you can reproduce the experimental results of the speech separation task in the submitted paper.
The WSJ0-2mix dataset is used during the training phase.
Example training settings are for DPRNN 3-blocks.
Detailed hyperparameter settings are enumerated in the paper.

- Training with SML
~~~
python train_sml.py --gpu=0 --batch_size=4 ----alpha=0.001 ----confidence=-15
~~~

### Evaluation

~~~
python testing.py 
~~~

### Dataset
We use commercial WSJ0 dataset (https://catalog.ldc.upenn.edu/LDC93S6B) as basis.
We evaluated the SML approach on the speaker-independent speech separation task 
using the WSJ0-2mix dataset. This is the common two-speaker benchmark used for 
speech separation in recent years.

### Citation

~~~
@inproceedings{tan2022selective,
  title={Selective Mutual Learning: An Efficient Approach for Single Channel Speech Separation},
  author={Tan, Ha Minh and Vu, Duc-Quang and Lee, Chung-Ting and Li, Yung-Hui and Wang, Jia-Ching},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={3678--3682},
  year={2022},
  organization={IEEE}
}
~~~
