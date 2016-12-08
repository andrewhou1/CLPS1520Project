#!/usr/bin/env python2.7
import argparse
import os
import time

import numpy as np
import tensorflow as tf

from model import CNNModel
from model import restore_model
from preprocessing import read_object_classes, FROM_GAMES, DATASETS, save_labels_array
from train import run_model_iter


def test_model(sess, model, dataset_iter, layer, use_patches=False, patches_per_image=1000, gaussian_sigma=None,
               color_map=None, output_dir=None):
    total_accuracy = 0
    class_correct_counts = np.zeros(model.num_classes)
    class_total_counts = np.zeros(model.num_classes)
    i = 0

    def iter_model():
        return run_model_iter(sess, model, image, labels, is_training=False, use_patches=use_patches,
                              patches_per_image=patches_per_image, gaussian_sigma=gaussian_sigma)

    for image, labels, img_id in dataset_iter():
        i += 1
        start_time = time.time()
        accuracy = 0.0
        if use_patches:
            patch_size = model.PATCH_SIZE
            for ops, patch_labels in iter_model():
                logits1, logits2, _ = ops
                logits = logits1 if layer == 1 else logits2
                _, output_h, output_w, _ = logits.shape
                predicted_label = np.argmax(logits[0, output_h / 2, output_w / 2, :])
                true_label = patch_labels[patch_size / 2, patch_size / 2]

                class_total_counts[true_label] += 1
                if true_label == predicted_label:
                    class_correct_counts[true_label] += 1
                    accuracy += 1
            print "Image #%d: %s Accuracy: %f (time: %.1fs)" % (
                i, img_id, accuracy / patches_per_image, time.time() - start_time)
        else:
            for logits1, logits2, _ in iter_model():
                logits = logits1 if layer == 1 else logits2
                stride = 4 if layer == 2 else 2
                predicted_labels = np.argmax(logits[0], axis=2)
                true_labels = labels[::stride, ::stride]

                correct_labels = np.equal(predicted_labels, true_labels)
                accuracy = np.mean(correct_labels)
                total_accuracy += accuracy

                for c in range(model.num_classes):
                    current_class_labels = np.equal(true_labels, c)
                    class_total_counts[c] += np.sum(current_class_labels)
                    class_correct_counts[c] += np.sum(np.equal(true_labels, c) * correct_labels)

                print "Image #%d: %s: Accuracy: %f (time: %.1fs)" % (
                    i, img_id, accuracy, time.time() - start_time)

        if output_dir is not None and color_map is not None:
            for layer_num in [1,2]:
                output_filename = os.path.join(output_dir, img_id + '_test_%d.png' % layer_num)
                predicted_labels = None
                for logits1, logits2, _ in iter_model():
                    logits = [logits1, logits2][layer_num - 1]
                    predicted_labels = np.argmax(logits[0], axis=2)
                predicted_labels = np.kron(predicted_labels, np.ones(shape=[4, 4]))
                save_labels_array(predicted_labels.astype(np.uint8), output_filename, colors=color_map)

    print "%d Images, Total Accuracy: %f" % (i, total_accuracy / i)
    print "Per Class correct counts:", class_correct_counts
    print "Per Class totals:", class_total_counts
    print "Per Class accuracy:", class_correct_counts / class_total_counts


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
    parser.add_argument('--use_patches', action='store_true', default=False,
                        help='Whether to evaluate model on individual patches')
    parser.add_argument('--patches_per_image', type=int, default=2000,
                        help='Number of patches to sample from each test image. Not used by default.')
    parser.add_argument('--gaussian_sigma', type=int, choices=[15, 30], default=None,
                        help='Size of gaussian mask to apply to patches. Not used by default.')
    parser.add_argument('--test_fraction', type=float, default=-0.2,
                        help='Fraction of data to test on. If positive, tests on first X images, otherwise tests on '
                             'last X images.')
    parser.add_argument('--layer', choices=[1,2], type=int, default=2,
                        help='Number of rCNN layers to use.')
    args = parser.parse_args()

    # load class labels
    category_colors, category_names, names_to_ids = read_object_classes(args.category_map)
    num_classes = len(category_names)

    # load dataset
    def dataset_func(): return DATASETS[args.dataset](args.data_dir, train_fraction=args.test_fraction)

    # TODO only test?
    # TODO don't hardcode these (maybe store them in config file?)
    model = CNNModel(25, 50, 1, num_classes, 1e-4, num_layers=2)

    sess = tf.Session()
    restore_model(sess, args.model)

    test_model(sess, model, dataset_func, args.layer, use_patches=args.use_patches, patches_per_image=args.patches_per_image,
               gaussian_sigma=args.gaussian_sigma, output_dir=args.output_dir, color_map=category_colors)


if __name__ == '__main__':
    main()
