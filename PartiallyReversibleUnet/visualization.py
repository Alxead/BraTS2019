"""
File: visualization.py
Author: VDeamoV
Email: vincent.duan95@outlook.com
Github: https://github.com/VDeamoV
Description:
    Partially copy from cp-vton source code
    Use function below to viusalize the image in the tensorboard
"""
import os

import cv2
import torch
import numpy as np


def tensor_for_board(img_tensor):
    """
    input:
        (n, 1, W, H) / (n, 3, W, H)
    function:
        (1, 1, W, H) -> (1, 3, W, H)
    """
    tensor = (img_tensor.clone()+1) * 0.5
    tensor.cpu().clamp(0, 1)

    # if tensor.size(1) == 1:
    #     tensor = tensor.repeat(1, 3, 1, 1)
    # else:
    #     tensor = tensor[:, [2, 1, 0], :, :]

    return tensor


def tensor_list_for_board(img_tensors_list):
    """
    change batch of images into one image
    input:
        [[one batch], [two batch]]
    """

    grid_h = len(img_tensors_list)
    grid_w = max(len(img_tensors)  for img_tensors in img_tensors_list)

    batch_size, channel, height, width = tensor_for_board(img_tensors_list[0]).size()
    canvas_h = grid_h * height
    canvas_w = grid_w * width
    canvas = torch.FloatTensor(batch_size, channel, canvas_h, canvas_w).fill_(0.5)
    for i, img_tensors in enumerate(img_tensors_list):
        for j, img_tensor in enumerate(img_tensors):
            offset_h = i * height
            offset_w = j * width
            tensor = tensor_for_board(img_tensor)
            canvas[:, :, offset_h : offset_h + height, offset_w : offset_w + width].copy_(tensor)

    return canvas


def board_add_image(board, tag_name, img_tensor, step_count):
    """
    Use api .add_image to add image to tensorboard
    """
    tensor = tensor_for_board(img_tensor)

    for i, img in enumerate(tensor):
        board.add_image('%s/%03d' % (tag_name, i), img, step_count)


def board_add_images(board, tag_name, img_tensors_list, step_count):
    """
    Use api .add_images to add image to tensorboard
    """
    tensor = tensor_list_for_board(img_tensors_list)

    for i, img in enumerate(tensor):
        board.add_image('%s/%03d' % (tag_name, i), img, step_count)


def board_add_embeddings(board, global_step, embeddings, metedata, label_img=None):
    """
    use to draw tsne, pac
    """

    board.add_embedding(
        embeddings,
        metedata=metedata,
        label_img=label_img,
        global_step=global_step
    )


def board_add_scalars(board, step_count, path, scalars):
    """
    draw multi scalars in one figure
    """
    board.add_scalars(path, scalars, step_count)


def save_images(img_tensors, img_names, save_dir):
    """
    Save images locally
    """
    for img_tensor, img_name in zip(img_tensors, img_names):
        pos = img_name.rfind('/')
        img_name = img_name[pos+1:]
        tensor = (img_tensor.clone()+1)*0.5 * 255
        tensor = tensor.cpu().clamp(0, 255)

        array = tensor.numpy().astype('uint8')
        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)

        cv2.imwrite(os.path.join(save_dir, img_name), array)

def test():
    """
    Test function for file
    """
    one_channel_fake_img = torch.Tensor(np.arange(224*224).reshape((1, 1, 224, 224)))
    normal_fake_img = torch.Tensor(np.arange(3*224*224).reshape((1, 3, 224, 224)))
    batch_img = [[one_channel_fake_img, normal_fake_img],[]]
    tensor_convert = tensor_for_board(one_channel_fake_img)
    assert tensor_convert.shape[1] == 3
    tensor_list = tensor_list_for_board(batch_img)
    __import__('ipdb').set_trace()


if __name__ == "__main__":
    test()