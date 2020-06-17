# Style Transfer in Videos
Repository for Deep Learning Final Project - UPF 2019/20<br>
Javier Rando Ramírez<br>
Eduard Vergés Franch<br>
Marcel Closes Pagan<br>


The main goal of this project is implementing style transfer using Neural Networks on videos. As a starting point, the [PyTorch implementation for style transfer on images](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html) has been used.
To use this approach on videos, every frame will be considered as a single image to which we must perform style transfer.

However, independent treatment of frames result on inconsistency between frames as many research has already pointed out [[1]](https://arxiv.org/abs/1807.01197).

To solve this issue, we have tested two different approaches to increase consistency between consecutive frames. More information about these approaches are found in `slides.pdf`. They are briefly summed up in the next two sections.

## Previous frame loss - Stabilizer 1
In this first approach that can be executed using the argument `--stabilizer 1`, we add a new term to the loss function. It will compute the mean square difference between the previous stylized frame and the current one.
This way, we can penalize difference between two consecutive stylized frames. The weight of this loss term cannot be very high since it would at some point prevent difference between frames.

## Optical Flow - Stabilizer 2
To improve stabilizer 1 limitation for moving objects we implemented an approach using Optical Flow. First of all, we compute the flow of moving objects between frames.
Then, these flows are used to compute a prediction of the current frame using the previous styled one. Like this, we avoid minimizing the difference with previous position of
objects and remove artifacts.

## Execution
There are two different scripts in this repository. The first, will generate the styled frames and the second one will generate a video from them with
the desired frame rate. This last one will be explained in **generate video** section.

To generate the styled frames we will need an input video and some style image. It is important that they have the same proportions.

This is a sample execution:
```
python main.py --video data/input/video1.mp4 --style data/input/colorful-style.jpg --outpath data/output --stabilizer 1 --style_weight 100000 --content_weight 1 --num_steps 150 --previous_weight 1 --output_filename video1_col_st1_pw1_2_
```

The arguments used can be changed to match your needs:
* `--video`: path to the input video.
* `--style`: path to the style image.
* `--outpath`: path to the folder where styled frames will be stored.
* `--stabilizer`: method for time stabilization. Options are: `0` which uses independent styling, `1` where difference with previous styled frame is used as constraint, `2` the difference is now computed between the current frame and the previous styled one with
optical flow applied to predict the current styled frame. (More information about stabilizers can be found in the slides).
* `--style_weight`: default 1000000. Weight for the style image in the loss.
* `--content_weight`: default 1. Weight for the current original frame in the loss.
* `--style_weight`: default 1. Weight for the previous styled frame. Only applies for stabilizers 1 and 2.
* `--num_steps`: default 200. Number of epochs for each frame.
* `--output_filename`: prefix for the output frames. An index will be appended at the end. Therefore, we recommend using a name that doesn't end with a number. You will use this name for the generate video script.


## Generate video

A video can be generated using `create_video.py` script. Sample execution:

```
python3 create_video.py -i ./data/output/frames/ -if video1_col_st1_pw1_2_ -o ./data/output/videos -of video1_col_st1_pw1_2_ -fp 10
```

The arguments used can be changed to match your needs:
* `-i`: folder where frames are stored. Matches `--outpath` in previous step.
* `-if`: input frames prefix which matches `--output_filename` argument in previous step.
* `-o`: path to the folder where the video will be stored.
* `-of`: name for the output video.
* `-fp`: frames per second.
