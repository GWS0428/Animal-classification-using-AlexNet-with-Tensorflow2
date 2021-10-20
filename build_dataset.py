"""Split the Animal-10 dataset into train/dev/test and resize images to 227x227.

The Animal-10 dataset comes in the following format:
    butterfly/
        1.jpeg
        ...
    cat/
        942.jpg
        ...

Resizing to (227, 227) reduces the dataset size and loading smaller images makes training faster.

we'll take 80% of images as train set, 10% of images as validation set and 10% of images as test set.
"""

import argparse
import random
import os
import pathlib

from PIL import Image
from tqdm import tqdm


SIZE = 227

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/Animals', help="Directory with the Animals-10 dataset")
parser.add_argument('--output_dir', default='data/resized_Animals', help="Where to write the new data")


def resize_and_save(filename, output_dir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`
    
    Args:
        filename: (string) path to a file
        output_dir: (string) path to a directory to save a resized image
        size: (int) size after resizing an image
    """
    image = Image.open(filename)
    
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    image.save(os.path.join(output_dir, os.path.basename(filename)))


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    data_directory_names = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spyder', 'squirrel']
    data_directory_dict = {}
    for directory in data_directory_names:
        data_directory_path = os.path.join(args.data_dir, directory)
        data_directory_dict[directory] = data_directory_path

    # Get the filenames in each directory
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(123)
    filenames_dict = {}
    for directory_name, directory_path in data_directory_dict.items():
        filenames = os.listdir(directory_path)
        filenames = [ os.path.join(directory_path, f) for f in filenames ]
        filenames.sort()
        random.shuffle(filenames)
        filenames_dict[directory_name] = filenames

    # Split the images into 80% train, 10% dev and 10% test
    train_dict = {}
    dev_dict = {}
    test_dict = {}
    for directory_name, filenames in filenames_dict.items():
        split_1 = int(0.8 * len(filenames))
        split_2 = int(0.9 * len(filenames))
        
        train_filenames = filenames[:split_1]
        dev_filenames = filenames[split_1:split_2]
        test_filenames = filenames[split_2:]
        
        train_dict[directory_name] = train_filenames
        dev_dict[directory_name] = dev_filenames
        test_dict[directory_name] = test_filenames

    splitted_filenames = {'train': train_dict,
                          'dev': dev_dict,
                          'test': test_dict}
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Preprocess train, dev and test
    for split in ['train', 'dev', 'test']:
        
        output_dir_split = os.path.join(args.output_dir, split)
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        filename_dict = splitted_filenames[split]
        for directory_name, filenames in filename_dict.items():
            for filename in tqdm(filenames):
                resize_and_save(filename, os.path.join(output_dir_split, directory_name), size=SIZE)

    print("Done building dataset")
