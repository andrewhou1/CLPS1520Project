#!/usr/bin/env python2.7
import argparse
import os

from os.path import isfile

from model import CNNModel
import tensorflow as tf

from preprocessing import read_object_classes, labels_to_np_array, image_to_np_array


def train(model, train_files):
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    label_f, image_f = train_files[0]
    label_1 = labels_to_np_array(label_f)
    image_1 = image_to_np_array(image_f)

    height_1, width_1, _ = image_1.shape
    print "Image size: ", height_1, width_1

    # # extract a patch for testing
    # # in this case, label_path is a single number
    # label_patch = label_1[model.patch_size/2, model.patch_size/2]
    # # image_patch is a 3D array
    # image_patch = image_1[:model.patch_size, :model.patch_size, :]

    i = 0
    skip_num = 200
    for img in range(10):
        error_for_image = 0
        for y0 in range(height_1 - model.patch_size):
            for x0 in range(width_1 - model.patch_size):
                i += 1
                if i % skip_num != 0:
                    continue
                # extract a patch for testing
                # in this case, label_path is a single number
                label_patch = label_1[y0 + model.patch_size / 2, x0 + model.patch_size / 2]
                # image_patch is a 3D array
                image_patch = image_1[y0:y0 + model.patch_size, x0:x0 + model.patch_size, :]

                feed_dict = {model.inpt: [image_patch], model.output: [[label_patch]]}
                error, _ = sess.run([model.error, model.train_step], feed_dict=feed_dict)
                error_for_image += error
                if i % (skip_num * 2000) == 0:
                    print "Training error (%d): %f" % (i, error)
        print "Average image error:", error_for_image * skip_num / i


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
    args = parser.parse_args()

    # load class labels
    category_colors, cols_to_ids, category_names, names_to_ids = read_object_classes(args.category_map)

    labels_dir = os.path.join(args.data_dir, 'labels')
    images_dir = os.path.join(args.data_dir, 'images')

    # TODO get only image files
    labels = [os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if isfile(os.path.join(labels_dir, f))]
    images = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if isfile(os.path.join(images_dir, f))]
    train_files = zip(labels, images)
    model = CNNModel(args.hidden_size_1, args.hidden_size_2, args.patch_size, args.batch_size, len(category_names), args.learning_rate)
    train(model, train_files)
    pass

if __name__ == '__main__':
    main()
