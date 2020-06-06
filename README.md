# Style Transfer in Videos
Repository for Deep Learning Final Project - UPF 2019/20
Javier Rando Ramírez
Eduard Vergés Franch
Marcel Closes Pagan


The main goal of this project is implementing style transfer using Neural Networks on videos. As a starting point, the [PyTorch implementation for style transfer on images](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html) has been used.
To use this approach on videos, every frame will be considered as a single image to which we must perform style transfer.

However, independent treatment of frames result on inconsistency between frames as many research has already pointed out [[1]](https://arxiv.org/abs/1807.01197).

To solve this issue, we have tested two different approaches to increase consistency between consecutive frames.

## Previous frame loss
In this first approach that can be executed using the argument `--stabilizer 1`, we add a new term to the loss function. It will compute the mean square difference between the previous stylized frame and the current one.
This way, we can penalize difference between two consecutive stylized frames. The weight of this loss term cannot be very high since it would at some point prevent difference between frames.

## Optical Flow
To be documented.

## Execution


To execute use
```
 python3 main.py --video ./data/input/video-short.mp4 --style ./data/input/video-style.jpg --outpath ./data/output -s 0
```

Also there are other optional arguments
```
--num_steps
--content_weight
--style_weight
--previous_weight
```