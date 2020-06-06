import argparse
import cv2
import os
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def create_video (input_folder,
                  input_filename,
                  output_folder,
                  output_filename,
                  fps):

    frames = []
    files = [f for f in os.listdir(input_folder) if f.startswith(input_filename)]
    format = os.path.splitext(files[0])[1]
    files = [os.path.splitext(f)[0] for f in files]
    files.sort(key=natural_keys)

    print('Reading frames...')
    for f in files:
        img = cv2.imread(os.path.join(input_folder, f+format))
        height, width, layers = img.shape
        size = (width, height)
        frames.append(img)

    print("{} frames were read.".format(len(frames)))

    out = cv2.VideoWriter(os.path.join(output_folder, output_filename+'.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    print('Writing frames...')
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()

    print('Process finished!')

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video")
    parser.add_argument("-i", "--input_folder", type=str, required=True)
    parser.add_argument("-if", "--input_filename", type=str, required=True)
    parser.add_argument("-o", "--output_folder", type=str, required=True)
    parser.add_argument("-of", "--output_filename", type=str, required=True)
    parser.add_argument("-fp", "--fps", type=int, required=True)

    args = parser.parse_args()
    input_folder = args.input_folder
    input_filename = args.input_filename
    output_folder = args.output_folder
    output_filename = args.output_filename
    fps = args.fps

    # Run main
    create_video(input_folder, input_filename, output_folder, output_filename, fps)