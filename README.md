# MNIST-with-PYGAD

MNIST classification with pygad optimized CNNs.

### Install 
[pytorch](https://pytorch.org/)

pip3 install pandas

pip3 install pygad

### Quickstart
1.
###### Train a small population for a short while, using pygad default settings

python train_GA.py --config configs/default_settings_small_experiment.json


2.
###### Train for a short while, using pygad default settings. Evaluating on the complete trainingset of 50000 images to get better fitness estimate. Initialize weights closer to zero in order to make network easier to train.

python train_GA.py --config configs/evaluation_on_complete_trainingset_better_initialization.json

3.
###### Train for 10000 generations. Evaluating on the complete trainingset of 50000 images to get better fitness estimate. Initialize weights closer to zero in order to make network easier to train.

python train_GA.py --config configs/10000_generations_ss.json
