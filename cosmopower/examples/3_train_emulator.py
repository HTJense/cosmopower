#!/usr/bin/env/python

# Author: A. Spurio Mancini, H. T. Jense

import os, glob
import numpy as np
import cosmopower as cp
import tensorflow as tf
from cosmopower import cosmopower_NN
from cosmopower.parser import YAMLParser
from cosmopower.dataset import Dataset

"""
In the previous file, we created a training dataset for a linear P(k,z) emulator.
Here, we will train the emulator over this dataset.
"""
parser = YAMLParser("example.yaml")

network_settings = parser.settings("Pk/lin")

print(network_settings)

"""
We assume that we simply want to train a dense neural network.
There is also the `3a_train_PCA_emulator.py` file which does the same as this
file but trains a PCA+NN type emulator.
"""
input_file = os.path.join(parser.path, "spectra", "Pk_lin.hdf5")
output_file = os.path.join(parser.path, "networks", parser.network_filename("Pk/lin"))

print("Training the linear P(k) emulator.")
print(f"Input files: {input_file}")
print(f"Output file: {output_file}")

settings = parser.get_traits("Pk/lin")

datasets = [Dataset(parser, "Pk/lin", "Pk_lin.hdf5")]
validation = [Dataset(parser, "Pk/lin", "Pk_lin_validation.hdf5")]

network = cosmopower_NN(parameters=parser.network_input_parameters("Pk/lin"),
                        modes=parser.modes("Pk/lin"),
                        verbose=True,
                        trainable=True,
                        **settings.get("n_traits", {}))

with tf.device("/device:CPU:0"):
    network.train(training_data=datasets,
                  filename_saved_model=output_file,
                  validation=validation,
                  **parser.network_training_parameters("Pk/lin"))
