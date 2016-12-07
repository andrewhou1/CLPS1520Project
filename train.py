#!/usr/bin/env python2.7
import argparse
import random
import time

import numpy as np
import tensorflow as tf

from model import CNNModel, save_model, restore_model
from preprocessing import read_object_classes, DATASETS, FROM_GAMES, get_patch, gaussian


def run_model_iter(sess, model, image, labels, is_training=False, use_patches=False, patches_per_image=1000,
                   gaussian_sigma=None):
    if is_training:
        # For training, only run loss and train ops
        ops_to_run = [model.loss, model.train_step]
    else:
        # For testing, get outputs of both layers as well as loss
        ops_to_run = [model.logits[0], model.logits[1], model.loss]
    i = 0

    h, w = labels.shape

    if use_patches:
        patch_size = model.PATCH_SIZE
        if gaussian_sigma is not None:
            mask = gaussian(g_sigma=gaussian_sigma, g_size=patch_size)
        else:
            mask = 1
        for _ in range(patches_per_image):
            y = random.random() * (h - 2 * patch_size) + patch_size
            x = random.random() * (w - 2 * patch_size) + patch_size
            patch = get_patch(image, center=(y, x), patch_size=patch_size) * mask
            patch_labels = get_patch(labels, center=(y, x), patch_size=patch_size)
            input_patch = np.append(patch, np.zeros(shape=[patch_size, patch_size, model.num_classes],
                                                    dtype=np.float32), axis=2)
            feed_dict = {model.inpt: [input_patch], model.output: [patch_labels]}
            ops_results = sess.run(ops_to_run, feed_dict=feed_dict)
            yield ops_results, patch_labels
    else:
        input_image = np.append(image, np.zeros(shape=[h, w, model.num_classes], dtype=np.float32), axis=2)
        feed_dict = {model.inpt: [input_image], model.output: [labels]}
        yield sess.run(ops_to_run, feed_dict=feed_dict)


def train(sess, model, dataset_iter, num_epochs, use_patches=False, patches_per_image=1000, gaussian_sigma=None,
          save_path=None):
    def iter_model():
        return run_model_iter(sess, model, image, labels, is_training=True, use_patches=use_patches,
                              patches_per_image=patches_per_image, gaussian_sigma=gaussian_sigma)

    for i in range(num_epochs):
        print 'Running epoch %d/%d...' % (i + 1, num_epochs)
        n = 0
        for image, labels, img_id in dataset_iter():
            start_time = time.time()
            n += 1
            if use_patches:
                losses = [ops[0] for ops, _ in iter_model()]
            else:
                losses = [loss for loss, _ in iter_model()]

            avg_loss = sum(losses) / len(losses)
            elapsed_time = time.time() - start_time
            print "Trained on image #%d (%s): Loss: %f Elapsed time: %.1f" % (n, img_id, avg_loss, elapsed_time)

        if save_path is not None:
            print "Epoch %i finished, saving trained model to %s..." % (i + 1, save_path)
            save_model(sess, save_path)


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='An rCNN scene labeling model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default=FROM_GAMES, choices=DATASETS.keys(),
                        help='Type of dataset to use. This determines the expected format of the data directory')
    parser.add_argument('--data_dir', type=str, help='Directory for image and label data')
    parser.add_argument('--category_map', type=str, help='File that maps colors ')
    parser.add_argument('--hidden_size_1', type=int, default=25, help='First Hidden size for CNN model')
    parser.add_argument('--hidden_size_2', type=int, default=50, help='Second Hidden size for CNN model')
    parser.add_argument('--patch_size', type=int, default=67, help='Patch size for input images')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training CNN model')
    # TODO figure out batch size
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training CNN model')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs for training CNN model')
    parser.add_argument('--use_patches', action='store_true', default=False,
                        help='Whether to train model on individual patches')
    parser.add_argument('--patches_per_image', type=int, default=1000,
                        help='Number of patches to sample for each image during training of CNN model')
    parser.add_argument('--gaussian_sigma', type=int, choices=[15, 30], default=None,
                        help='Size of gaussian mask to apply to patches. Not used by default.')
    parser.add_argument('--fix_random_seed', action='store_true', default=False,
                        help='Whether to reset random seed at start, for debugging.')
    parser.add_argument('--model_save_path', type=str, default=None,
                        help='Optional location to store saved model in.')
    parser.add_argument('--model_load_path', type=str, default=None,
                        help='Optional location to load saved model from.')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='If true, only trains on one image, to test the training code quickly.')
    parser.add_argument('--train_fraction', type=float, default=0.8,
                        help='Fraction of data to train on. If positive, trains on first X images, otherwise trains on '
                             'last X images.')

    args = parser.parse_args()

    if args.fix_random_seed:
        random.seed(0)

    # load class labels
    category_colors, category_names, names_to_ids = read_object_classes(args.category_map)
    num_classes = len(category_names)

    # create function that when called, provides iterator to an epoch of the data
    dataset_func = DATASETS[args.dataset]
    if args.dry_run:
        def dataset_epoch_iter():
            return dataset_func(args.data_dir, num_train=1)
    else:
        def dataset_epoch_iter():
            return dataset_func(args.data_dir, train_fraction=args.train_fraction)

    model = CNNModel(args.hidden_size_1, args.hidden_size_2, args.batch_size, num_classes,
                     args.learning_rate, num_layers=2)

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    if args.model_load_path is not None:
        restore_model(sess, args.model_load_path)
    train(sess, model, dataset_epoch_iter, num_epochs=args.num_epochs, use_patches=args.use_patches,
          patches_per_image=args.patches_per_image, save_path=args.model_save_path, gaussian_sigma=args.gaussian_sigma)

    print "Saving trained model to %s ..." % args.model_save_path
    save_model(sess, args.model_save_path)


if __name__ == '__main__':
    main()
