#!/bin/bash
#SBATCH -J StyleTransfer
#SBATCH --chdir==/shared/home/u150399/deep_learning_2020
#SBATCH -o /shared/home/u150399/deep_learning_2020/logs/slurm.%N.%J.%u.out # STDOUT
#SBATCH -e /shared/home/u150399/deep_learning_2020/logs/slurm.%N.%J.%u.err # STDERR
#SBATCH --exclusive="user"

source /shared/profiles.d/easybuild.sh

ml Cython/0.29.10-foss-2019b-Python-3.6.6
ml torchvision/0.2.1-foss-2019b-Python-3.6.6-PyTorch-1.1.0
ml numpy/1.18.1-foss-2019b-Python-3.6.6
ml OpenCV/3.4.7-foss-2019b-Python-3.6.6
ml matplotlib/3.0.3-foss-2019b-Python-3.6.6
ml PyTorch/1.1.0-foss-2019b-Python-3.6.6-CUDA-9.0.176
ml TensorFlow/1.10.1-foss-2019b-GPU-Python-3.6.6


python /shared/home/u150399/deep_learning_2020/main.py --video /shared/home/u150399/deep_learning_2020/data/input/video1.mp4 --style /shared/home/u150399/deep_learning_2020/data/input/colorful-style.jpg --outpath /shared/home/u150399/deep_learning_2020/data/output --stabilizer 1 --style_weight 100000 --content_weight 1 --num_steps 150 --previous_weight 1 --output_filename video1_col_st1_pw1_2
python /shared/home/u150399/deep_learning_2020/main.py --video /shared/home/u150399/deep_learning_2020/data/input/video2.mp4 --style /shared/home/u150399/deep_learning_2020/data/input/default-style.jpg --outpath /shared/home/u150399/deep_learning_2020/data/output --stabilizer 1 --style_weight 100000 --content_weight 2 --num_steps 150 --previous_weight 1 --output_filename video2_def_st1_pw2_2