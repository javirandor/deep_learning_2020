import cv2
import numpy as np

def generate_next_frame(prev_frame,
                        flow):
    h = flow.shape[0]
    w = flow.shape[1]
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]

    return cv2.remap(prev_frame, flow, None, cv2.INTER_LINEAR)

def generate_flows (frames):

    flows = []

    # Take first frame
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale=0.5, levels=5, winsize=11, iterations=5, poly_n=5, poly_sigma=1.1, flags=0)
        flows.append(flow)
        prev_gray = gray

    return flows


def read_frames(video_path):
    vc = cv2.VideoCapture(video_path)
    frames = []

    while (vc.isOpened()):
        ret, frame = vc.read()
        if not ret:
            break
        frames.append(frame)

    return frames