import torch.nn as nn
from constants import device, content_layers_default, style_layers_default, previous_layers_default
import copy
import torch.optim as optim
from loss import StyleLoss, ContentLoss, PreviousLoss
from utils import store_frame
from optical_flow import generate_next_frame


# ================================== NORMALIZATION ================================== #


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# ================================== BUILD MODEL ================================== #

def get_style_model_and_losses(cnn, normalization_mean, normalization_std, stabilizer,
                               style_img, content_img, previous_styled_img=None,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default,
                               previous_layers=previous_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of losses
    content_losses = []
    style_losses = []
    previous_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

        if name in previous_layers and stabilizer == 1 and previous_styled_img is not None:
            target = model(previous_styled_img).detach()
            previous_loss = PreviousLoss(target)
            model.add_module("previous_loss_{}".format(i), previous_loss)
            previous_losses.append(previous_loss)

    # now we trim off the layers after the last content and style losses
    if stabilizer == 0:
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]
        return model, style_losses, content_losses

    elif stabilizer == 1:
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss) or isinstance(model[i], PreviousLoss):
                break

        model = model[:(i + 1)]
        return model, style_losses, content_losses, previous_losses


# ================================== OPTIMIZER ================================== #

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


# ================================== STYLE TRANSFER ================================== #

def run_style_transfer_no_st(cnn, normalization_mean, normalization_std,
                             video_frames, style_img, input_frames, output_path, output_filename,
                             num_steps, style_weight, content_weight):

    """Run the style transfer without stabilizer"""

    resulting_frames = []

    for original_frame, input_frame, index in zip(video_frames, input_frames, [i for i in range(len(video_frames))]):

        print("RUNNING TRANSFER FOR FRAME {}/{}".format(index, len(video_frames)))

        model, style_losses, content_losses = get_style_model_and_losses(cnn=cnn,
                                                                         normalization_mean=normalization_mean,
                                                                         normalization_std=normalization_std,
                                                                         stabilizer=0,
                                                                         style_img=style_img,
                                                                         content_img=original_frame)
        optimizer = get_input_optimizer(input_frame)

        run = [0]
        while run[0] <= num_steps:

            def closure():
                # correct the values of updated input image
                input_frame.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_frame)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        input_frame.data.clamp_(0, 1)
        store_frame(output_path, index, output_filename, input_frame)  # Store frame
        resulting_frames.append(input_frame)

    return resulting_frames


def run_style_transfer_st(stabilizer, cnn, normalization_mean, normalization_std,
                           video_frames, style_img, input_frames, output_path, output_filename,
                           num_steps, style_weight, content_weight, previous_weight, flows=None):

    """Run the style transfer with the first stabilizer"""

    resulting_frames = []

    for original_frame, input_frame, index in zip(video_frames, input_frames, [i for i in range(len(video_frames))]):

        print("RUNNING TRANSFER FOR FRAME {}/{}".format(index, len(video_frames)))

        if index == 0:
            previous_styled_img = None
        else:
            previous_styled_img = resulting_frames[index - 1]

            if stabilizer == 2:
                previous_styled_img = generate_next_frame(previous_styled_img, flows[index - 1])

        model, style_losses, content_losses, previous_losses = get_style_model_and_losses(cnn=cnn,
                                                                                          normalization_mean=normalization_mean,
                                                                                          normalization_std=normalization_std,
                                                                                          stabilizer=1,
                                                                                          style_img=style_img,
                                                                                          content_img=original_frame,
                                                                                          previous_styled_img=previous_styled_img)
        optimizer = get_input_optimizer(input_frame)

        run = [0]
        while run[0] <= num_steps:

            def closure():
                # correct the values of updated input image
                input_frame.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_frame)
                style_score = 0
                content_score = 0
                previous_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                if index != 0:
                    for pl in previous_losses:
                        previous_score += pl.loss

                style_score *= style_weight
                content_score *= content_weight

                if index != 0:
                    previous_score *= previous_weight
                    loss = style_score + content_score + previous_score
                    loss.backward()

                else:
                    loss = style_score + content_score
                    loss.backward()

                run[0] += 1

                if run[0] % 50 == 0 and index != 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f} Flow Loss: {:4f}'.format(
                        style_score.item(), content_score.item(), previous_score.item()))
                    print()
                elif run[0] % 50 == 0 and index == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(style_score.item(), content_score.item()))
                    print()

                return style_score + content_score + previous_score

            optimizer.step(closure)

        # a last correction...
        input_frame.data.clamp_(0, 1)
        store_frame(output_path, index, output_filename, input_frame)  # Store frame
        resulting_frames.append(input_frame)

    return resulting_frames
