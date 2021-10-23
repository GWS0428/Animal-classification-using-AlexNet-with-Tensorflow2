"""General utility functions"""

import os
import json
import logging


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_history_to_json(h, json_path):
    """Saves dict of list of floats in json file

    This function will be used to store history data after training to json_path

    Args:
        h: (dict) of list of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        h = {k: [float(n) for n in v] for k, v in h.items()}
        json.dump(h, f, indent=4)


def find_next_path(base_path, base_name):
    """Find a next path name that follows a numbered sequence.
    
    path will be created like {base_name}_{number}.
    The smallest number that is not used yet is chosen to name a path
    
    Args:  
        base_path: (string) absolute path to directory of 'base_name'
        base_name: (string) base name of a path
    """
    i = 0
    while os.path.exists(os.path.join(base_path, base_name + '{}'.format(i))):
        i += 1
    return os.path.join(base_path, base_name + '{}'.format(i))

def path_to_next_json(base_path, base_name):
    """Find a path to next file that follows a numbered sequence.
    
    file will be created like {base_name}_{number}.
    The smallest number that is not used yet is chosen to name next file
    
    Args:  
        base_path: (string) absolute path to file of 'base_name'
        base_name: (string) base name of a file
    """
    i = 0
    while os.path.exists(os.path.join(base_path, base_name + '{}.json'.format(i))):
        i += 1
    return os.path.join(base_path, base_name + '{}.json'.format(i))