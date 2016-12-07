"""
Code for preprocessing and loading image and label data.
"""

import sys

import numpy as np
from PIL import Image
import os

from os.path import isfile


def read_object_classes(classes_map_filename):
    """
    Reads an index of object classes and their corresponding names and colors.
    Each line of the file has 5 elements: R,G,B values as floats, an integer ID, and a name as a string.
    :param classes_map_filename: The filename storing the index
    :return: a tuple of 4 items:
        1. an array of ID -> category color as RGB tuple (in [0, 255])
        2. a dictionary of category color (as an RGB tuple) -> ID
        3. an array of ID -> category name
        2. a dictionary of category name -> ID
    """
    # TODO handle different potential formats better
    format_description = "Each line should contain 5 elements: (float R, float G, float B, int ID, str Name)."
    ids = set()
    ids_to_cols = {}
    ids_to_names = {}
    names_to_ids = {}
    with open(classes_map_filename, 'r') as classes_file:
        for line in classes_file:
            try:
                vals = line.split()
                if len(vals) == 0:
                    continue
                elif len(vals) == 2:
                    has_cols = False
                    category_num = int(vals[0])
                    category_name = vals[1]
                elif len(vals) == 5:
                    has_cols = True
                    rgb = tuple([int(255 * float(s)) for s in vals[:3]])
                    category_num = int(vals[3])
                    category_name = vals[4]
                else:
                    raise ValueError("Category map must have either 2 or 5 columns")

                # check for duplicate categories
                if category_num in ids:
                    sys.stderr.write("A category with this number (%d) already exists.\n" % category_num)
                    continue
                if category_name in names_to_ids:
                    sys.stderr.write("A category with this name (%s) already exists.\n" % category_name)
                    continue

                ids.add(category_num)
                ids_to_names[category_num] = category_name
                names_to_ids[category_name] = category_num
                if has_cols:
                    ids_to_cols[category_num] = rgb

            except (ValueError, IndexError) as e:
                sys.stderr.write("%s %s\n" % (format_description, e))
                continue

    max_id = max(ids)
    category_colors = [None] * (max_id + 1)
    category_names = [None] * (max_id + 1)
    for cat_id in ids:
        category_names[cat_id] = ids_to_names[cat_id]
        if has_cols:
            category_colors[cat_id] = ids_to_cols[cat_id]

    return category_colors, category_names, names_to_ids


def image_to_np_array(img_filename, float_cols=True):
    """
    Reads an image into a numpy array, with shape [height x width x 3]
    Each pixel is represented by 3 RGB values, either as floats in [0, 1] or as ints in [0, 255]
    :param img_filename: The filename of the image to load
    :param float_cols: Whether to load colors as floats in [0, 1] or as ints in [0, 255]
    :return: A numpy array containing the image data
    """
    img = Image.open(img_filename)
    img.load()
    if float_cols:
        data = np.asarray(img, dtype="float32") / 255.0
    else:
        data = np.asarray(img, dtype="uint8")
    return data


def labels_to_np_array(lab_filename):
    """
    Reads an image of category labels as a numpy array of category IDs.
    NOTE: The image data must already be in a color pallette such that color # corresponds to label ID.
    The "Playing for Data" dataset is configured in this way (http://download.visinf.tu-darmstadt.de/data/from_games/)
    :param lab_filename: The filename of the label image to load
    :return: A numpy array containing the label ID for each pixel
    """
    img = Image.open(lab_filename)
    img.load()
    data = np.asarray(img, dtype="uint8")
    return data


def text_labels_to_np_array(lab_filename):
    label_file = open(lab_filename, 'r')
    # TODO right now were just ignoring negative ("unknown") labels. Need a nicer way to do this in long term
    labels = [map(lambda n: max(0, int(n)), l.split()) for l in label_file.readlines()]
    return np.array(labels, dtype=np.int8)


def save_labels_array(labels, output_filename, colors):
    """
    Saves a numpy array of labels to an paletted image.
    :param colors: An array of colors for each index. Should correspond to label ID's in 'labels'
    :param labels: A 2D array of labels
    :param output_filename: The filename of the image to output
    """
    img = Image.fromarray(obj=labels, mode="P")
    # palette is a flattened array of r,g,b values, repreesnting the colors in the palette in order.
    palette = []
    for c in colors:
        palette.extend(c)
    img.putpalette(palette)
    img.save(output_filename)


def get_patch(array, center, patch_size):
    """
    Returns a square 2D patch of an array with a given size and center. Also returns other dimensions of the array,
    uncropped.
    NOTE: does not do bounds checking.
    :param array: A numpy array
    :param center: The coordinates of the center, as a list or array of length 2
    :param patch_size: A single number representing the width and height of the patch.
    :return: A square patch of the image with the given center and size.
    """
    rounded_width = int(patch_size / 2)
    return array[center[0] - rounded_width: center[0] + rounded_width + 1,
                 center[1] - rounded_width: center[1] + rounded_width + 1]


def from_games_dataset(data_dir, train_fraction=None, num_train=None):
    labels_dir = os.path.join(data_dir, 'labels')
    images_dir = os.path.join(data_dir, 'images')

    # TODO get only image files
    labels = [os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if
              isfile(os.path.join(labels_dir, f)) and not f.startswith('.')]
    labels = sorted(labels)
    images = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if
              isfile(os.path.join(images_dir, f)) and not f.startswith('.')]
    images = sorted(images)
    train_files = zip(labels, images)

    # if specified, only choose subset of training data
    if train_fraction is not None and num_train is None:
        num_train = int(len(train_files) * train_fraction)
    if num_train is not None:
        train_files = train_files[:num_train]

    for label_f, image_f in train_files:
        print "Current image:", os.path.basename(image_f)
        if os.path.basename(label_f) != os.path.basename(image_f):
            print "UNEQUAL IMAGE NAMES!"
        image = image_to_np_array(image_f)
        labels = labels_to_np_array(label_f)
        yield image, labels


# TODO negative label nums could mess up paletted output
def stanford_bgrounds_dataset(data_dir, train_fraction=None, num_train=None):
    labels_dir = os.path.join(data_dir, 'labels')
    images_dir = os.path.join(data_dir, 'images')

    # TODO get only image files
    labels = [os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if
              isfile(os.path.join(labels_dir, f)) and not f.startswith('.') and f.endswith('.regions.txt')]
    labels = sorted(labels)
    images = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if
              isfile(os.path.join(images_dir, f)) and not f.startswith('.')]
    images = sorted(images)
    train_files = zip(labels, images)

    # if specified, only choose subset of training data
    if train_fraction is not None and num_train is None:
        num_train = int(len(train_files) * train_fraction)
    if num_train is not None:
        train_files = train_files[num_train:]

    for label_f, image_f in train_files:
        if os.path.basename(label_f).split('.')[0] != os.path.basename(image_f).split('.')[0]:
            print "UNEQUAL IMAGE NAMES!", label_f, image_f
        img_id = os.path.basename(label_f).split('.')[0]
        image = image_to_np_array(image_f)
        labels = text_labels_to_np_array(label_f)
        yield image, labels, img_id


def gaussian(g_sigma, g_size):
    """
    Creates a 2D gaussian mask with values form 0 to 1, of the given size and variance.
    :param gSigma: Filter size
    :param g_Size: Patch size
    :return: A gaussian filter of the given size and variance
    """
    x1 = np.linspace(-g_size / 2, (g_size / 2) - 1, g_size)
    y1 = np.linspace(-g_size / 2, (g_size / 2) - 1, g_size)

    mx, my = np.meshgrid(x1, y1)

    g_window = np.exp(-(mx ** 2 + my ** 2) / (2 * g_sigma ** 2))

    return g_window


# list of datasets for which we have iterators
FROM_GAMES = 'from-games'
SIFT_FLOW = 'sift-flow'
STANFORD_BGROUND = 'stanford-bground'
DATASETS = {FROM_GAMES: from_games_dataset, SIFT_FLOW: None, STANFORD_BGROUND: stanford_bgrounds_dataset}


def main():
    infile = sys.argv[1]
    image = image_to_np_array(infile, float_cols=False)
    for gaussian_sigma in [15, 30]:
        mask = gaussian(g_sigma=gaussian_sigma, g_size=image.shape[0])
        mask = np.expand_dims(mask, axis=2)
        mask = np.repeat(mask, repeats=3, axis=2)


        masked_image = image * mask
        print masked_image
        output_image = Image.fromarray(masked_image.astype(dtype=np.uint8), mode="RGB")
        output_image.save(infile + str(gaussian_sigma) + '.png')


if __name__ == '__main__':
    main()
