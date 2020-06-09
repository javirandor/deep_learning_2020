import cv2
import numpy as np
from PIL import Image
from constants import loader, device, unloader
import torch


def generate_next_frame(prev_frame,
                        flow):
    new_frames = None

    h = flow.shape[0]
    w = flow.shape[1]
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]

    prev_frame = prev_frame.cpu().clone()  # we clone the tensor to not do changes on it
    prev_frame = prev_frame.squeeze(0)  # remove the fake batch dimension
    prev_frame = unloader(prev_frame)
    prev_frame = np.float32(prev_frame)

    frame_pred = cv2.remap(prev_frame, flow, None, cv2.INTER_LINEAR)

    image = Image.fromarray((frame_pred * 255).astype(np.uint8))
    image = loader(image).unsqueeze(0)

    new_frames = image.to(device, torch.float)
    return new_frames


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
        f = Image.fromarray(frame)
        f = f.resize((227, 128), Image.ANTIALIAS)
        f = loader(f).squeeze(0)
        f = f.to(device, torch.float)
        f = f.cpu().detach().numpy()
        f = np.moveaxis(f, 0, -1)
        f = np.uint8(f)

        frames.append(f)
    vc.release()
    return frames
