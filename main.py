import argparse
from utils import video_loader, image_loader, generate_input_frames
import torchvision.models as models
from constants import device, cnn_normalization_std, cnn_normalization_mean
from model import run_style_transfer_no_st, run_style_transfer_st
from optical_flow import read_frames, generate_flows


def main(in_video: str,
         style_img: str,
         output_path: str,
         stabilizer: int,
         num_steps: int,
         style_weight: int,
         content_weight: int,
         previous_weight: float,
         output_filename: str):

    print("Style Weight = {}".format(style_weight))
    # Load video
    video_frames = video_loader(in_video)

    # Load style image
    style_image = image_loader(style_img)

    # Check they have same size
    assert style_image.size() == video_frames[0].size(), \
        "Input video and style image must have the same dimensions"

    flows = None

    if (stabilizer == 2):  ## If we are using optical flow, generate flow between frames
        in_frames = read_frames(in_video)
        flows = generate_flows(in_frames)

    # Load pre-trained VGG model
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # Generate input frames
    input_frames = generate_input_frames(video_frames)

    # Run Style Transfer
    if stabilizer == 0:
        print('Running style_transfer_nost')
        transformed_frames = run_style_transfer_no_st(cnn=cnn,
                                                      normalization_mean=cnn_normalization_mean,
                                                      normalization_std=cnn_normalization_std,
                                                      video_frames=video_frames,
                                                      style_img=style_image,
                                                      input_frames=input_frames,
                                                      num_steps=num_steps,
                                                      style_weight=style_weight,
                                                      content_weight=content_weight,
                                                      output_path=output_path,
                                                      output_filename=output_filename)

    elif stabilizer == 1 or stabilizer == 2:
        print('Running style_transfer_st')
        transformed_frames = run_style_transfer_st(stabilizer=stabilizer,
                                                   cnn=cnn,
                                                   normalization_mean=cnn_normalization_mean,
                                                   normalization_std=cnn_normalization_std,
                                                   video_frames=video_frames,
                                                   style_img=style_image,
                                                   input_frames=input_frames,
                                                   num_steps=num_steps,
                                                   style_weight=style_weight,
                                                   content_weight=content_weight,
                                                   previous_weight=previous_weight,
                                                   output_path=output_path,
                                                   output_filename=output_filename,
                                                   flows=flows)

    print("Style video transfer successfully completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Style Transfer")
    parser.add_argument("-i", "--video", type=str, required=True)
    parser.add_argument("-st", "--style", type=str, required=True)
    parser.add_argument("-o", "--outpath", type=str, required=True)
    parser.add_argument("-s", "--stabilizer", type=int, required=False, choices=[0, 1, 2], default=0)
    parser.add_argument("-ns", "--num_steps", type=int, required=False, default=200)
    parser.add_argument("-cw", "--content_weight", type=int, required=False, default=1)
    parser.add_argument("-sw", "--style_weight", type=int, required=False, default=1000000)
    parser.add_argument("-pw", "--previous_weight", type=float, required=False, default=0.5)
    parser.add_argument("-of", "--output_filename", type=str, required=False, default='frame')

    args = parser.parse_args()
    input_video = args.video
    style_img = args.style
    output_path = args.outpath
    stabilizer = args.stabilizer
    num_steps = args.num_steps
    content_weight = args.content_weight
    style_weight = args.style_weight
    previous_weight = args.previous_weight
    output_file = args.output_filename

    # Run main
    main(input_video, style_img, output_path, stabilizer, num_steps, style_weight, content_weight, previous_weight, output_file)
