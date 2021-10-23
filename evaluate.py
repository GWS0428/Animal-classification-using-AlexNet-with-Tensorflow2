"""Evaluate the model"""

import argparse
import logging
import os

import tensorflow as tf

from model.model_fn import build_model
from model.utils import Params, save_dict_to_json
from model.utils import set_logger


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default=os.path.join('experiments', 'base_model'),
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default=os.path.join('data', 'resized_Animals'),
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=os.path.join('experiments', 'base_model', 'weights_checkpoint0'),
                    help="Subdirectory of model dir containing the weights")


if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.random.set_seed(123)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    data_dir = args.data_dir
    test_data_dir = os.path.join(data_dir, "test")

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(227, 227),
        batch_size=params.batch_size,
        class_mode='categorical')

    # Define the model
    logging.info("Creating the model...")
    model = build_model(params)

    # Reload weights from directory if specified
    assert os.path.exists(args.restore_from), "{} directory for restoring is not found.".format(args.restore_from)
    logging.info("Restoring parameters from {}".format(args.restore_from))
    if os.path.isdir(args.restore_from):
        latest = tf.train.latest_checkpoint(args.restore_from)
        model.load_weights(latest)

    logging.info("Starting test")
    results = model.evaluate_generator(test_generator, verbose=1)
    results_dict = {'loss': results[0], 'accuracy': results[1]}
    save_path = os.path.join(args.model_dir, "test_result.json")
    save_dict_to_json(results_dict, save_path)
    logging.info("End of test")