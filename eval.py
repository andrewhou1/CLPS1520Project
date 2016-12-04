#!/usr/bin/env python2.7
import argparse
import os

import numpy as np
import tensorflow as tf

from model import CNNModel
from model import restore_model
from preprocessing import read_object_classes, image_to_np_array, labels_to_np_array, get_patch, save_labels_array


def test_model(sess, model, images, labels, patch_size, output_dir=None, category_colors=None):
    """
    Tests the model on the given images and labels
    :param patch_size:
    :param sess: The tensorflow session within which to run the model
    :param model: An rCNN model
    :param images: A series of image filenames
    :param labels: A series of label filenames (corresponding images/labels should be in the same order)
    :param output_dir: An (optional) directory in which to store predicted labels as images.
    :param category_colors: A mapping of category index to color, to create images
    """
    for image_f, label_f in zip(images, labels):
        print "Testing on image %s..." % image_f
        image = image_to_np_array(image_f, float_cols=True)
        labels = labels_to_np_array(label_f)
        h, w, _ = image.shape
        predicted_labels = np.zeros([h, w], dtype=np.uint8)
        pixels_correct = 0
        error_for_image = 0
        i = 0
        for y in range(patch_size, h - patch_size):
            # # for debug, only do first 10K
            # if i > 1e4:
            #     break

            for x in range(patch_size, w - patch_size):
                i += 1
                input_image = get_patch(image, (y, x), patch_size)
                input_image = np.append(input_image,
                                        np.zeros(shape=[patch_size, patch_size, 1], dtype=np.float32),
                                        axis=2)
                input_label = labels[y, x]
                feed_dict = {model.inpt: [input_image], model.output: [[input_label]]}

                error, logits = sess.run([model.error, model.logits], feed_dict=feed_dict)
                error_for_image += error
                output_label = np.argmax(logits)
                if output_label == input_label:
                    pixels_correct += 1
                predicted_labels[y, x] = output_label

                if i % 1000 == 0:
                    print "%d/%d pixels done..." % (i, (h - 2 * patch_size) * (w - 2 * patch_size))

        print "Tested on image %s: Accuracy is %.2f%%, error per pixel is %f." % (
            image_f, (100.0 * pixels_correct) / i, error_for_image / i)
        if output_dir is not None:
            if category_colors is None:
                raise ValueError("Color index not provided, can't output images.")
            output_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(label_f))[0] + '_test.png')
            print "output: ", output_filename
            save_labels_array(predicted_labels, output_filename=output_filename, colors=category_colors)


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate an rCNN scene labelling model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, help='Filename of saved model')
    parser.add_argument('--category_map', type=str, help='File that maps colors ')
    parser.add_argument('--images', type=str, nargs='+', help='Filename of test image')
    parser.add_argument('--labels', type=str, nargs='+', help='Filename of test labels')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to store model output. By default no output is generated.')
    args = parser.parse_args()

    # load class labels
    category_colors, category_names, names_to_ids = read_object_classes(args.category_map)
    num_classes = len(category_names)
    # TODO don't hardcode these (maybe store them in config file?)
    model = CNNModel(25, 50, 1, num_classes, 1e-4, num_layers=2)

    sess = tf.Session()
    restore_model(sess, args.model)

    test_model(sess, model, args.images, args.labels, patch_size=23, output_dir=args.output_dir,
               category_colors=category_colors)


if __name__ == '__main__':
    main()
