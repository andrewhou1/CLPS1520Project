"""
Code for preprocessing and loading image and label data.
"""

import sys

import numpy as np
from PIL import Image


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
                rgb = tuple([int(255 * float(s)) for s in vals[:3]])
                category_num = int(vals[3])
                category_name = vals[4]

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
                ids_to_cols[category_num] = rgb

            except (ValueError, IndexError) as e:
                sys.stderr.write("%s %s\n" % (format_description, e))
                continue

    max_id = max(ids)
    category_colors = [None] * (max_id + 1)
    category_names = [None] * (max_id + 1)
    for cat_id in ids:
        category_names[cat_id] = ids_to_names[cat_id]
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
    rounded_width = patch_size // 2
    return array[center[0] - rounded_width: center[0] + rounded_width + 1,
                 center[1] - rounded_width: center[1] + rounded_width + 1]
