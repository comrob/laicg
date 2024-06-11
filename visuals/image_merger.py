import os

from PIL import Image
from typing import List
DEFAULT_WIDTH = 30
DEFAULT_HEIGHT = 30

def horizontal_merge_images(images: List[Image.Image]) -> Image.Image:
    width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT
    for im in images:
        if im is not None:
            width, height = im.size

    total_width = width * len(images)
    max_height = height

    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        if im is not None:
            new_im.paste(im, (x_offset, 0))
        x_offset += width

    return new_im


def vertical_merge_images(images: List[Image.Image]) -> Image.Image:
    width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT
    for im in images:
        if im is not None:
            width, height = im.size

    total_width = width
    max_height = height * len(images)

    new_im = Image.new('RGB', (total_width, max_height))

    y_offset = 0
    for im in images:
        if im is not None:
            new_im.paste(im, (0, y_offset))
        y_offset += height
    return new_im


def grid_merge_portfolio(portfolio_directories: List[List[str]], output_directory, prefix=""):
    pics = []

    for row_dir_pth in portfolio_directories:
        for dir_pth in row_dir_pth:
            for pic_name in os.listdir(dir_pth):
                pics.append(pic_name)
    pics = list(set(pics))

    for pic_name in pics:
        out_pth = os.path.join(output_directory, prefix + "_" + pic_name)
        row_imgs = []
        for row_dir_pth in portfolio_directories:
            imgs = []
            for dir_pth in row_dir_pth:
                pth = os.path.join(dir_pth, pic_name)
                if os.path.exists(pth):
                    imgs.append(Image.open(pth))
                else:
                    imgs.append(None)
            row_imgs.append(horizontal_merge_images(imgs))
        new_im = vertical_merge_images(row_imgs)
        new_im.save(out_pth)


