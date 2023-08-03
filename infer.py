import candle
import os
from precily_baseline_keras import main
import json
import tensorflow as tf
from tensorflow import keras


# This should be set outside as a user environment variable
os.environ['CANDLE_DATA_DIR'] = '/homes/ac.tfeng/git/Precily'
file_path = os.path.dirname(os.path.realpath(__file__))

additional_definitions = None
required = None


class Precily_candle(candle.Benchmark):
    def set_locals(self):
        """
        Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the benchmark.
        """
        if required is not None: 
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definisions = additional_definitions

def initialize_parameters():

    # Build benchmark object
    preprocessor_bmk = Precily_candle(
        file_path,
        'precily_default_model.txt',
        "keras",
        prog="Precily_candle",
        desc="Data Preprocessor",
    )

    # Initialize parameters
    gParameters = candle.finalize_parameters(preprocessor_bmk)

    return gParameters

def preprocess(params):
    # Model-specific changes
    params['test_data'] = os.environ['CANDLE_DATA_DIR'] + 'Fig1/Fig1c/Fig1c_Precily_pathways/' + params['test_data']
    return params


def run(params):
    trained_model = os.environ['CANDLE_DATA_DIR'] + 'Fig1/Fig1c/Fig1c_Precily_pathways/' + "precily_cv" + ".hdf5"
    model = keras.models.load_model(trained_model)