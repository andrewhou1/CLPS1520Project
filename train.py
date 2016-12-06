#!/usr/bin/env python2.7
import argparse
import random
import time

import numpy as np
import tensorflow as tf

from model import CNNModel, save_model
from preprocessing import read_object_classes, DATASETS, FROM_GAMES


def train(sess, model, dataset_iter, num_epochs, patch_size, patches_per_image=1000, save_path=None):
    for i in range(num_epochs):
        print 'Running epoch %d/%d...' % (i + 1, num_epochs)
        for image, labels in dataset_iter():
            for row in labels:
                for l in row:
                    if l < 0 or l >= model.num_classes:
                        print "INVALID label:", l

            start_time = time.time()
            h, w, _ = image.shape

            input_image = np.append(image, np.zeros(shape=[h, w, model.num_classes], dtype=np.float32), axis=2)
            feed_dict = {model.inpt: [input_image], model.output: [labels]}
            loss, _ = sess.run([model.loss, model.train_step], feed_dict=feed_dict)
            print "Average error for this image: %f (time: %.1fs)" % (loss, time.time() - start_time)

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
    parser.add_argument('--patches_per_image', type=int, default=1000,
                        help='Number of patches to sample for each image during training of CNN model')
    parser.add_argument('--fix_random_seed', action='store_true', default=False,
                        help='Whether to reset random seed at start, for debugging.')
    parser.add_argument('--model_save_path', type=str, default=None,
                        help='Optional location to store saved model in.')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='If true, only trains on one image, to test the training code quickly.')

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
            return dataset_func(args.data_dir, train_fraction=0.8)

    model = CNNModel(args.hidden_size_1, args.hidden_size_2, args.batch_size, num_classes,
                     args.learning_rate, num_layers=2)

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    train(sess, model, dataset_epoch_iter, patch_size=args.patch_size, num_epochs=args.num_epochs,
          patches_per_image=args.patches_per_image,
          save_path=args.model_save_path)

    print "Saving trained model to %s ..." % args.model_save_path
    save_model(sess, args.model_save_path)


if __name__ == '__main__':
    main()
