import pandas as pd
import sys
import re
import shutil
import os
import csv
import datetime
import numpy as np
import tensorflow as tf


def save_model(model, modelName):
    saved_network = model.to_json()
    name = (modelName + ".json").strip()
    with open(name, "w") as json_file:
        json_file.write(saved_network)
    model.save_weights(modelName)


def load_model(modelPath_json, modelPath_weights, loss_func, opt):
    """Used to load TF.Keras model
    @Params
    modelPath_json: Path to json
    modelPath_weights: Path to weights
    loss_func: Keras loss function object
    opt: Optimizer (e.g "adam" or "rmprop")"""
    json_file = open(
        modelPath_json, 'r')
    loaded_network_json = json_file.read()
    json_file.close()
    loaded_network = tf.keras.models.model_from_json(loaded_network_json)
    loaded_network.load_weights(modelPath_weights).expect_partial()
    loaded_network.compile(optimizer=opt,
                           loss=loss_func,
                           metrics=['mae'])
    return loaded_network
