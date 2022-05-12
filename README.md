# The Autoembedder
## Introduction
The Autoembedder is an autoencoder with additional embedding layers for the categorical columns. Its usage is flexible, and hyperparameters like the number of layers can be easily adjusted and tuned. Although primarily designed for Panda's dataframes, it can be easily modified to support other data structures.

## Let's get started
`training.py` is where everything begins. The following arguments can / should be set:

| Argument                           | Type  | Required | Default value    | Comment                                                                                                                                                                                  |
| ---------------------------------- | ----- | -------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| batch_size                         | int   | False    | 32               |                                                                                                                                                                                          |
| drop_last                          | int   | False    | 1                | True/False                                                                                                                                                                               |
| pin_memory                         | int   | False    | 1                | True/False                                                                                                                                                                               |
| num_workers                        | int   | False    | 0                | 0 means that the data will be loaded in the main process                                                                                                                                 |
| model_title                        | str   | False    | autoembedder_{datetime}.bin |                                                                                                                                                                                          |
| model_save_path                    | str   | False    |                  |                                                                                                                                                                                          |
| n_save_checkpoints                 | int   | False    |                  |                                                                                                                                                                                          |
| lr                                 | float | False    | 0.001            |                                                                                                                                                                                          |
| epochs                             | int   | True     |                  |
| layer_bias                             | int   | False     |  1                | True/False|                                                                                                                                                                                          |
| weight_decay                       | float | False    | 0                |                                                                                                                                                                                          |
| l1_lambda                          | float | False    | 0                |                                                                                                                                                                                          |
| xavier_init                        | int   | False    | 0                | True/False                                                                                                                                                                               |
| tensorboard_log_path               | str   | False    |                  |                                                                                                                                                                                          |
| train_input_path                   | str   | True     |                  |                                                                                                                                                                                          |
| test_input_path                    | str   | True     |                  |                                                                                                                                                                                                                                                                                                                                                                                    |
| activation_for_code_layer          | int   | False    | 0                | True/False, should the layer have an activation                                                                                                                                          |
| activation_for_final_decoder_layer | int   | False    | 0                | True/False, should the final decoder layer have an activation                                                                                                                            |
| hidden_layer_representation        | str   | True     |                  | Contains a string representation of a list of list of integers which represents the hidden layer structure. E.g.: `"[[64, 32], [32, 16], [16, 8]]"` activation                           |
| cat_columns                        | str   | False    | "[]"             | Contains a string representation of a list of list of categorical columns (strings). The columns which use the same encoder should be together in a list. E.g.: `"[['a', 'b'], ['c']]"`. |

So, something like this would do it:

```
$ python3 training.py --epochs 20 \
--train_input_path "path/to/your/train_data" \
--test_input_path "path/to/your/test_data" \
--hidden_layer_representation "[[12, 6], [6, 3]]"
```


## Why additional embedding layers?
The additional embedding layers automatically embed all columns with the Pandas `category` data type. If categorical columns have another data type, they will not be embedded and will be handled like the continuous columns. Simply encoding the categorical values (e.g., with the usage of a label encoder) decreases the quality of the outcome.
