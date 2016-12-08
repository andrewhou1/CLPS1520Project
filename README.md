# CLPS 1520 Project - Scene Labeling with Recurrent Convolutional Neural Networks

## Overview
This project implements a "Recurrent Convolutional Neural Network" (rCNN) for scene labeling,
as seen in [Pinheiro et al, 2014](http://www.jmlr.org/proceedings/papers/v32/pinheiro14.pdf).

### Summary of files
    README.md           -- README file
    accuracy_results/   -- Text files containing results of different models
    category_maps/      -- Text files containing category info for Stanford  & "Data from Games" datasets
    eval.py             -- Script for testing the model
    model.py            -- Code for rCNN model
    models/             -- Saved models from training
    preprocessing.py    -- Code for processing input data and image patches
    requirements.txt    -- Lists python package requirements
    test_results/       -- Model output, as images
    train.py            -- Script for trainin model

## Installation
 1. Clone or download this repository to your computer:

    ```git clone https://github.com/NP-coder/CLPS1520Project.git```

 2. Install the necessary Python requirements through `pip`:

    ```pip install -r requirements.txt```

    This project only requires Tensorflow version 11, and PIL for image manipulation.

 3. Download a dataset with which to use the model.
 We used the [Stanford Background Dataset](http://dags.stanford.edu/projects/scenedataset.html) which has 700+ 320x240 images and 8 classes.
 The code also works with the [Data from Games](https://download.visinf.tu-darmstadt.de/data/from_games/)
 dataset, which has 25,000 1914 × 1052 with 38 categories.

    To use another dataset, make sure it is organized similarly to one of the above two, and specify while training and testing which dataset it is "mimicking".
  Specifically, both datasets had the data in one folder, with subfolders "labels" and "images" for labels and images, respectively.
  The stanford dataset had labels in the format of space-separated digits in a text file, while the "Data from  Games" dataset had labels in the form of paletted images,
   where each color corresponds to a different label.

  4. Generate a text file that maps colors to category numbers nad labels. Each line of the file has five space-separated values:


      R  G  B category_num category_id

   Category files for the Stanford Background and Data From Games datasets are provided in the folder `category_maps`.

## Running

### Training
For training, use the `train.py` script. The following command trains the model on the Stanford dataset:

    python train.py --dataset stanford-bground --category_map category_maps/stanford_bground_categories.txt --data_dir <STANFORD DATA FOLDER> --model_save_path <SAVED_MODEL_FILE>


Running `train.py -h` will show additional parameters for the script, including different hyperparameters.
`train.py` also supports the ability to train on specific patches surrounding individual pixels,
and to apply gaussian filters to the input patches.

### Testing
For testing, use the `eval.py` script.
This script will get per-class accuracies for each image, as well as output predicted labels as image files.
The following command loads a saved model and evaluates accuracy on the stanford data set:

    python eval.py --model <SAVED_MODEL> --category_map category_maps/stanford_bground_categories.txt --dataset stanford-bground --data_dir <STANFORD DATA FOLDER> --output_dir <RESULT IMAGES FOLDER>


Similar to `train.py`, use the `-h` flag to see more parameters.

## Credits

 - Tyler Barnes-Diana
 - Zhenhao Hou (Andrew)
 - Jae Hwan Hwang (James)
 - Raphael Kargon
