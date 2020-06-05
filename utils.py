from PIL import Image
from constants import loader, device, unloader
import torch
import cv2
import os
import matplotlib.pyplot as plt


def image_loader(image_path: str):
    image = Image.open(image_path)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def video_loader(video_path: str):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frames = []
    while success:
        image = Image.fromarray(image)  # Convert to PIL image
        image = loader(image).unsqueeze(0)
        frames.append(image.to(device, torch.float))
        success, image = vidcap.read()
    return frames


def imshow(tensor, title=None):
    plt.figure()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show()


def generate_input_frames(video_frames: list):
    input_frames = []
    for frame in video_frames:
        input_frames.append(frame.clone())
        # if you want to use white noise instead uncomment the below line:
        # input_frames.append(torch.randn(frame.data.size(), device=device))


def store_frames (output_path: str,
                  frames: list,
                  output_frame_name: str = "frame"):

    # Check if frames folder exists. If not, create it.
    if not os.path.exists(os.path.join(output_path, 'frames')):
        os.makedirs(os.path.join(output_path, 'frames'))

    count = 1

    for frame in frames:
        image = frame.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)  # remove the fake batch dimension
        image = unloader(image).convert('RGB')
        plt.imshow(image)
        image.save(output_path + 'frames/{}{}.jpg'.format(output_frame_name, count))
        count += 1