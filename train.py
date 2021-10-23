"""Train the model"""

import argparse
import logging
import os
import random
import pathlib
import datetime

import tensorflow as tf

from model.utils import Params
from model.utils import set_logger
from model.utils import save_history_to_json
from model.utils import find_next_path
from model.utils import path_to_next_json
from model.model_fn import build_model


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default=os.path.join('experiments', 'base_model'),
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default=os.path.join('data', 'resized_Animals'),
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.random.set_seed(123)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    model_dir_has_weight_folder = os.path.isdir(
        os.path.join(args.model_dir, "weight_checkpoints"))
    overwritting = model_dir_has_weight_folder and args.restore_from is None
    assert not overwritting, "Weight checkpoints found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    train_data_dir = os.path.join(data_dir, "train")
    dev_data_dir = os.path.join(data_dir, "dev")

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(227, 227),
        batch_size=params.batch_size,
        class_mode='categorical')
    validation_generator = validation_datagen.flow_from_directory(
        dev_data_dir,
        target_size=(227, 227),
        batch_size=params.batch_size,
        class_mode='categorical')

    # Define the model
    logging.info("Creating the model...")
    model = build_model(params)
  
    # Reload weights from directory if specified
    # directory for checkpoint and weights must be renamed to prevent overwriting
    if args.restore_from is not None:
        logging.info("Restoring parameters from {}".format(args.restore_from))
        assert os.path.isdir(args.restore_from), "{} directory for restoring is not found, aborting to avoid overwrite".format(args.restore_from)
        latest = tf.train.latest_checkpoint(args.restore_from)
        model.load_weights(latest)
            
    # Create a ModelCheckpoint callback to save weights of the model
    checkpoint_path = os.path.join('weights_checkpoint0', 'cp-{epoch:04d}.ckpt')
    checkpoint_dir = os.path.join(args.model_dir, checkpoint_path)
    
    if args.restore_from is not None:
        checkpoint_path = find_next_path(os.path.realpath(args.model_dir), 'weights_checkpoint')
        checkpoint_dir = os.path.join(checkpoint_path, 'cp-{epoch:04d}.ckpt')

    TRAIN_SIZE = train_generator.n
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                                     save_weights_only=True,
                                                     save_freq='epoch',
                                                     verbose=1,
                                                     period=5)
    
    # Create a tensorboard callback to save log
    log_dir = os.path.join(args.model_dir, "logs0")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    if args.restore_from is not None:
        log_dir = find_next_path(os.path.realpath(args.model_dir), 'logs')
        log_dir = os.path.join(log_dir, 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = validation_generator.n // validation_generator.batch_size
    history = model.fit_generator(generator=train_generator,
                                steps_per_epoch=STEP_SIZE_TRAIN,
                                validation_data=validation_generator,
                                validation_steps=STEP_SIZE_VALID,
                                epochs=params.num_epochs,
                                verbose=1,
                                callbacks=[cp_callback, tensorboard_callback])

    save_path = os.path.join(args.model_dir, "training_result0.json")
    if args.restore_from is not None:
        save_path = path_to_next_json(os.path.realpath(args.model_dir), 'training_result')
    save_history_to_json(history.history, save_path)

    logging.info("End of training for {} epoch(s)".format(params.num_epochs))
