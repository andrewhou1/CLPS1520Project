#!/usr/bin/env python2.7
import argparse
import os

import numpy as np
import tensorflow as tf
import time

from model import CNNModel
from model import restore_model
from preprocessing import read_object_classes, image_to_np_array, labels_to_np_array, get_patch, save_labels_array, \
    FROM_GAMES, DATASETS


def test_model(sess, model, dataset_iter, color_map, output_dir):
    total_accuracy = 0
    class_correct_counts = np.zeros(model.num_classes)
    class_total_counts = np.zeros(model.num_classes)
    i = 0
    for image, labels in dataset_iter():
        i += 1
        start_time = time.time()
        h, w, _ = image.shape

        input_image = np.append(image, np.zeros(shape=[h, w, model.num_classes], dtype=np.float32), axis=2)
        feed_dict = {model.inpt: [input_image], model.output: [labels]}
        logits, error = sess.run([model.logits[1], model.loss], feed_dict=feed_dict)
        predicted_labels = np.argmax(logits[0], axis=2)
        true_labels = labels[::4, ::4]

        correct_labels = np.equal(predicted_labels, true_labels)
        accuracy = np.mean(correct_labels)
        total_accuracy += accuracy

        for c in range(model.num_classes):
            current_class_labels = np.equal(true_labels, c)
            class_total_counts[c] += np.sum(current_class_labels)
            class_correct_counts[c] += np.sum(np.equal(true_labels, c) * correct_labels)

        print "Error: %f Accuracy: %f (time: %.1fs)" % (error, accuracy, time.time() - start_time)

    print "%d Images, Total Accuracy: %f" % (i, total_accuracy/i)
    print "Per Class accuracy:", class_correct_counts / class_total_counts
    print np.sum(class_correct_counts / class_total_counts)


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate an rCNN scene labelling model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, help='Filename of saved model')
    parser.add_argument('--category_map', type=str, help='File that maps colors ')
    parser.add_argument('--dataset', type=str, default=FROM_GAMES, choices=DATASETS.keys(),
                        help='Type of dataset to use. This determines the expected format of the data directory')
    parser.add_argument('--data_dir', type=str, help='Directory for image and label data')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to store model output. By default no output is generated.')
    parser.add_argument('--patch_size', type=int, default=67, help='Size of input patches')
    args = parser.parse_args()

    # load class labels
    category_colors, category_names, names_to_ids = read_object_classes(args.category_map)
    num_classes = len(category_names)

    # load dataset
    def dataset_func(): return DATASETS[args.dataset](args.data_dir)

    # TODO don't hardcode these (maybe store them in config file?)
    model = CNNModel(25, 50, 1, num_classes, 1e-4, num_layers=2)

    sess = tf.Session()
    restore_model(sess, args.model)

    test_model(sess, model, dataset_func)


if __name__ == '__main__':
    main()
