# STAT5242-Project

## Setup

This repo contains code accompanying Fall 2021 STAT5242 Final Project. To get set up, make sure you are using Python <3.10and  simply follow the two-step process

1. Clone this repo and install requirements using

```
pip install -r requirements.txt
```

2. Download the G-Research data from <https://www.kaggle.com/c/g-research-crypto-forecasting/data> and place the unzipped folder in the 'data' directory

## Training models

All code is held in the 'src' folder with figures and model checkpoints saved to 'figures' and 'checkpoints', respectively. The code is written to run from being called in the base directory. The training script takes 4 command-line arguments, namely:

1. training_mode: which model to train, options are lstm, gcn, additive, or sequential
2. technicals_config: file name of json file mapping technical indicators to the name of the function implementing said indicator in src/preprocessing/utils.py file
3. epochs: number of epochs to train for
4. model_name: name of model for saving training loss plot (saved to figures/{model_name}_training_loss.pdf) and model checkpoint (saved to checkpoints/{model_name}.pth)

For example, to train a vanilla LSTM model for 10 epochs, enter the following into the command- ine from the base directory of this repo

```
python src/train.py --training_mode lstm --technials_config src/technicals_config.json --epochs 10 --model_name test_lstm
```

## Evaluating models

The evaluation script is formated very similarly to the training script. It takes 3 command-line arguments, namely:

1. eval_model: which model to evaluate, options are lstm, gcn, additive, or sequential
2. technicals_config: file name of json file mapping technical indicators to the name of the function implementing said indicator in src/preprocessing/utils.py file (must be same used to train the model being loaded)
3. model_name: name of model to be loaded from checkpoints (loads checkpoints/{model_name}.pth) that contains the state_dict of the trained model

To evaluate the above trained vanilla LSTM model, use the following command line from the base directory of this repo

```
python src/eval.py --eval_model lstm --technials_config src/technicals_config.json --model_name test_lstm
```