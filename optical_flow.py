import cv2
import numpy as np
from PIL import Image
from constants import device, unloader
import torch
from constants import loader
from utils import store_frame


def generate_next_frame(prev_frame,
                        flow):

    # Previous frame to array
    prev_frame = unloader(prev_frame.squeeze().detach().cpu())
    prev_frame = np.uint8(prev_frame)

    h = flow.shape[0]
    w = flow.shape[1]
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]

    prediction = cv2.remap(prev_frame, flow, None, cv2.INTER_LINEAR)  # Type array

    # Prediction to tensor in CUDA
    prediction = loader(unloader(prediction))
    new_frame = prediction.unsqueeze(0).to(device)

    return new_frame


def generate_flows(frames):
    flows = []

    # Take first frame
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for frame, index in zip(frames[1:], [i for i in range(1, len(frames) + 1)]):
        # print(index)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale=0.5, levels=5, winsize=11, iterations=5, poly_n=5, poly_sigma=1.1, flags=0)
        prev_gray = gray

        flows.append(flow)

    return flows


def read_frames(video_path):
    vc = cv2.VideoCapture(video_path)
    frames = []

    while (vc.isOpened()):
        ret, frame = vc.read()
        if not ret:
            break
        frame = cv2.resize(frame, (910, 512))
        frames.append(frame)
    return frames
