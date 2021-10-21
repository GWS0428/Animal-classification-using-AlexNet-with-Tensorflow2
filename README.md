# Animal Classification using AlexNet with Tensorflow 2

_Authors: Wooseok Gwak_

This is the repository that implements Alexnet and trains it to classify Animals-10 dataset using tensorflow 2. Given an image of an animal, the model predict the correct label.


## Requirements

I recommends using python3 and conda virtual environment.

```
conda create -n myenv python=3.7
conda activate myenv
conda install requirements.txt
```

When you're done working on the project, deactivate the virtual environment with `conda deactivate`.


## Download the Animals-10 dataset

I used the Animals-10 dataset from kaggle. The dataset can be downloaded from [here](https://www.kaggle.com/alessiocorrado99/animals10).

If you want to train our model, download the Animals-10 dataset (~477 MB) containing photos of 10 kinds of animals to the /data folder. Otherwise, you can use our trained model without downloading dataset.

Here is the structure of the data after downloading dataset:

'''
raw-img/
    cane/
        1.jpeg
        ...
    cavallo/
        142.jpeg
        ...
    ...
'''

Since the folder names are written in Italian, you need to rename the folders in English based on the translate.txt file in /data.

Here is the structure of the data after renaming dataset:

```
Animals/
    butterfly/
        1.jpeg
        ...
    cat/
        142.jpeg
        ...
    ...
```


## Train model

1. **Build the dataset of size 227x227**: make sure you complete this step before training

The new reiszed dataset will be located by default in `data/resized_Animals`.

```bash
python build_dataset.py --data_dir data/Aniamls --output_dir data/resized_Animals
```

2. **set hyperparameters ( in experiment/ folder)** 

We created a `base_model` directory for you under the `experiments` directory. It countains a file `params.json` which sets the parameters for the experiment. It looks like

```json
{
    "learning_rate": 1e-3,
    "batch_size": 32,
    "num_epochs": 10,
    ...
}
```

For every new experiment, you will need to create a new directory under `experiments` with a similar `params.json` file.

3. **Train a model** Simply run

```bash
python train.py --data_dir data/Animals --model_dir experiments/base_model
```

It will instantiate a model and train it on the training set following the parameters specified in `params.json`.

4. **Display the results** 

'''bash
tensorboard --logdir experiments/base_model/logs0
'''

5. **Evaluation on the test set** 

Once you've run many experiments and selected your best model and hyperparameters based on the performance on the development set, you can finally evaluate the performance of your model on the test set. Run

```bash
python evaluate.py --data_dir data/Animals --model_dir experiments/base_model
```


## Resources

