#!/bin/bash
#SBATCH -J Test
#SBATCH --chdir==/shared/home/u150399/deep_learning_2020
#SBATCH -o /shared/home/u150399/deep_learning_2020/logs/slurm.%N.%J.%u.out # STDOUT
#SBATCH -e /shared/home/u150399/deep_learning_2020/logs/slurm.%N.%J.%u.err # STDERR


source /shared/profiles.d/easybuild.sh

ml Cython/0.29.10-foss-2019b-Python-3.6.6
ml torchvision/0.2.1-foss-2019b-Python-3.6.6-PyTorch-1.1.0
ml numpy/1.18.1-foss-2019b-Python-3.6.6
ml OpenCV/3.4.7-foss-2019b-Python-3.6.6
ml matplotlib/3.0.3-foss-2019b-Python-3.6.6
ml PyTorch/1.1.0-foss-2019b-Python-3.6.6-CUDA-9.0.176
ml TensorFlow/1.10.1-foss-2019b-GPU-Python-3.6.6

python /shared/home/u150399/deep_learning_2020/main.py --video /shared/home/u150399/deep_learning_2020/data/input/video-10fps-short.mp4 --style /shared/home/u150399/deep_learning_2020/data/input/default-style.jpg --outpath /shared/home/u150399/deep_learning_2020/data/output --stabilizer 0 --style_weight 500000 --content_weight 1 --num_steps 200 --output_filename frame_def_nost_
