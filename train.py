#!/usr/bin/env python2.7
import argparse
import os
import random

from os.path import isfile

from model import CNNModel
import tensorflow as tf

from preprocessing import read_object_classes, labels_to_np_array, image_to_np_array
from randomPixels import gencoordinates


def get_patch(array, center, patch_size):
    """
    Returns a square 2D patch of an array with a given size and center. Also returns other dimensions of the array,
    uncropped.
    NOTE: does not do bounds checking.
    :param array: A numpy array
    :param center: The coordinates of the center, as a list or array of length 2
    :param patch_size: A single number representing the width and height of the patch.
    :return:
    """
    rounded_width = patch_size // 2
    return array[center[0] - rounded_width: center[0] + rounded_width + 1,
                 center[1] - rounded_width: center[1] + rounded_width + 1]


def train(model, train_files, num_epochs, patches_per_image=1000):
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    for i in range(num_epochs):
        print 'Running epoch %d/%d...' % (i+1, num_epochs)
        for label_f, image_f in train_files:
            error_per_image = 0
            labels = labels_to_np_array(label_f)
            image = image_to_np_array(image_f)
            h, w, _ = image.shape
            coords_iter = gencoordinates(model.patch_size, h - model.patch_size - 1, model.patch_size,
                                         w - model.patch_size - 1)

            for _ in range(patches_per_image):
                patch_center = next(coords_iter)
                input_image = get_patch(image, patch_center, model.patch_size)
                input_label = labels[patch_center[0], patch_center[1]]
                feed_dict = {model.inpt: [input_image], model.output: [[input_label]]}
                error, _, logits = sess.run([model.error, model.train_step, model.logits], feed_dict=feed_dict)
                error_per_image += error

            print "Average error for this image (%s): %f" % (os.path.basename(image_f), error_per_image / patches_per_image)


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='An rCNN scene labeling model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, help='Directory for image and label data')
    parser.add_argument('--category_map', type=str, help='File that maps colors ')
    parser.add_argument('--hidden_size_1', type=int, default=25, help='First Hidden size for CNN model')
    parser.add_argument('--hidden_size_2', type=int, default=50, help='Second Hidden size for CNN model')
    parser.add_argument('--patch_size', type=int, default=23, help='Patch size for input images')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training CNN model')
    # TODO figure out batch size
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training CNN model')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs for training CNN model')
    parser.add_argument('--patches_per_image', type=int, default=1000,
                        help='Number of patches to sample for each image during training of CNN model')
    parser.add_argument('--fix_random_seed', action='store_true', default=False,
                        help='Whether to reset random seed at start, for debugging.')
    args = parser.parse_args()

    if args.fix_random_seed:
        random.seed(0)



    # load class labels
    category_colors, category_names, names_to_ids = read_object_classes(args.category_map)

    labels_dir = os.path.join(args.data_dir, 'labels')
    images_dir = os.path.join(args.data_dir, 'images')

    # TODO get only image files
    labels = [os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if isfile(os.path.join(labels_dir, f))]
    images = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if isfile(os.path.join(images_dir, f))]
    train_files = zip(labels, images)
    num_classes = len(category_names)
    print "*** NUM_CLASSES", num_classes
    model = CNNModel(args.hidden_size_1, args.hidden_size_2, args.patch_size, args.batch_size, num_classes,
                     args.learning_rate)
    train(model, train_files, num_epochs=args.num_epochs, patches_per_image=args.patches_per_image)
    pass


if __name__ == '__main__':
    main()
