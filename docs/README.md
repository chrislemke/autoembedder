![The autoembedder](https://raw.githubusercontent.com/chrislemke/autoembedder/master/docs/assets/images/image.png)
# The Autoembedder
[![deploy package](https://github.com/chrislemke/autoembedder/actions/workflows/deploy-package.yml/badge.svg)](https://github.com/chrislemke/autoembedder/actions/workflows/deploy-package.yml)
[![pypi](https://img.shields.io/pypi/v/autoembedder)](https://pypi.org/project/autoembedder/)
![python version](https://img.shields.io/pypi/pyversions/autoembedder?logo=python&logoColor=yellow)
[![docs](https://img.shields.io/badge/docs-mkdoks%20material-blue)](https://chrislemke.github.io/autoembedder/)
[![license](https://img.shields.io/github/license/chrislemke/autoembedder)](https://github.com/chrislemke/autoembedder/blob/main/LICENSE)
[![downloads](https://img.shields.io/pypi/dm/autoembedder)](https://pypistats.org/packages/autoembedder)
[![mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
## Introduction
The Autoembedder is an autoencoder with additional embedding layers for the categorical columns. Its usage is flexible, and hyperparameters like the number of layers can be easily adjusted and tuned. The data provided for training can be either a path to a [Dask](https://docs.dask.org/en/stable/dataframe.html) or [Pandas](https://pandas.pydata.org/) DataFrame stored in the Parquet format or the DataFrame object directly.

## Installation
If you are using [Poetry](https://python-poetry.org/), you can install the package with the following command:
```bash
poetry add autoembedder
```
If you are using [pip](https://pypi.org/project/pip/), you can install the package with the following command:
```bash
pip install autoembedder
```


## Installing dependencies
With [Poetry](https://python-poetry.org/):
```bash
poetry install
```
With [pip](https://pypi.org/project/pip/):
```bash
pip install -r requirements.txt
```
## Usage
### 0. Some imports
```python
from autoembedder import Autoembedder, dataloader, fit
```
### 1. Create dataloaders
First, we create two [`dataloaders`](https://chrislemke.github.io/autoembedder/autoembedder.data/#autoembedder.data.Dataset.__init__). One for training, and the other for validation data. As `source` they either accept a path to a Parquet file, to a folder of Parquet files or a [Pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)/[Dask](https://docs.dask.org/en/stable/dataframe.html) DataFrame.
```python
train_dl = dataloader(train_df)
valid_dl = dataloader(vaild_df)
```

### 2. Set parameters
Now, we need to set the parameters. They are going to be used for handling the data and training the model. In this example, only parameters for the training are set. [Here](https://github.com/chrislemke/autoembedder#parameters) you find a list of all possible parameters. This should do it:
```python
parameters = {
    "hidden_layers": [[25, 20], [20, 10]],
    "epochs": 10,
    "lr": 0.0001,
    "verbose": 1,
}
```

### 3. Initialize the autoembedder
Then, we need to initialize the [autoembedder](https://chrislemke.github.io/autoembedder/autoembedder.model/#autoembedder.model.Autoembedder). In this example, we are not using any categorical features. So we can skip the `embedding_sizes` argument.
```python
model = Autoembedder(parameters, num_cont_features=train_df.shape[1])
```

### 4. Train the model
Everything is set up. Now we can [fit](https://chrislemke.github.io/autoembedder/autoembedder.learner/#autoembedder.learner.fit) the model.
```python
fit(parameters, model, train_dl, valid_dl)
```

## Example
Check out [this Jupyter notebook](https://github.com/chrislemke/autoembedder/blob/main/example.ipynb) for an applied example using the [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.

## Parameters
This is a list of all parameters that can be passed to the Autoembedder for training:

| Argument                           | Type  | Required (only for running using the `training.py`) | Default value               | Comment                                                                                                                                                                                  |
| ---------------------------------- | ----- | --------------------------------------------------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| batch_size                         | int   | False                                               | 32                          |                                                                                                                                                                                          |
| drop_last                          | int   | False                                               | 1                           | True/False                                                                                                                                                                               |
| pin_memory                         | int   | False                                               | 1                           | True/False                                                                                                                                                                               |
| num_workers                        | int   | False                                               | 0                           | 0 means that the data will be loaded in the main process                                                                                                                                 |
| use_mps                            | int   | False                                               | 0                           | Set this to `1` if you want to use the [MPS Backend](https://pytorch.org/docs/master/notes/mps.html) for running on Mac using the M1 GPU. process                                        |
| model_title                        | str   | False                                               | autoembedder_{`datetime`}.bin |                                                                                                                                                                                          |
| model_save_path                    | str   | False                                               |                             |                                                                                                                                                                                          |
| n_save_checkpoints                 | int   | False                                               |                             |                                                                                                                                                                                          |
| lr                                 | float | False                                               | 0.001                       |                                                                                                                                                                                          |
| amsgrad                            | int   | False                                               | 0                           | True/False                                                                                                                                                                               |
| epochs                             | int   | True                                                |                             |
| dropout_rate                       | float | False                                               | 0                           | Dropout rate for the dropout layers in the encoder and decoder.                                                                                                                          |
| layer_bias                         | int   | False                                               | 1                           | True/False                                                                                                                                                                               |  |
| weight_decay                       | float | False                                               | 0                           |                                                                                                                                                                                          |
| l1_lambda                          | float | False                                               | 0                           |                                                                                                                                                                                          |
| xavier_init                        | int   | False                                               | 0                           | True/False
| activation                        | str   | False                                               | tanh                           | Activation function; either `tanh`, `relu`, `leaky_relu` or `elu`                                                                                                                                                                               |
| tensorboard_log_path               | str   | False                                               |                             |                                                                                                                                                                                          |
| trim_eval_errors                   | int   | False                                               | 0                           | Removes the max and min loss when calculating the `mean loss diff` and `median loss diff`. This can be useful if some rows create very high losses.                                      |
| verbose                            | int   | False                                               | 0                           | Set this to `1` if you want to see the model summary and the validation and evaluation results. set this to `2` if you want to see the training progress bar. `0` means no output.       |
| target                             | str   | False                                               |                             | The target column. If not set no evaluation will be performed.                                                                                                                           |
| train_input_path                   | str   | True                                                |                             |                                                                                                                                                                                          |
| test_input_path                    | str   | True                                                |                             |
| eval_input_path                    | False | True                                                |                             | Path to the evaluation data. If no path is provided no evaluation will be performed.                                                                                                     |                                                                                                                           |
| hidden_layer_representation        | str   | True                                                |                             | Contains a string representation of a list of list of integers which represents the hidden layer structure. E.g.: `"[[64, 32], [32, 16], [16, 8]]"` activation                           |
| cat_columns                        | str   | False                                               | "[]"                        | Contains a string representation of a list of list of categorical columns (strings). The columns which use the same encoder should be together in a list. E.g.: `"[['a', 'b'], ['c']]"`. |


## Run the training script
Something like this should do it:
```bash
python3 training.py --epochs 20 \
--train_input_path "path/to/your/train_data" \
--test_input_path "path/to/your/test_data" \
--hidden_layer_representation "[[12, 6], [6, 3]]"
```


## Why additional embedding layers?
The additional embedding layers automatically embed all columns with the Pandas `category` data type. If categorical columns have another data type, they will not be embedded and will be handled like continuous columns. Simply encoding the categorical values (e.g., with the usage of a label encoder) decreases the quality of the outcome.
