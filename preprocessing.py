import candle
import os
from precily_baseline_keras import main
import json


# This should be set outside as a user environment variable
#os.environ['CANDLE_DATA_DIR'] = '/homes/brettin/Singularity/workspace/data_dir/'
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
    fname=params['train_data']
    origin= os.environ['CANDLE_DATA_DIR'] + 'Fig1/Fig1c/Fig1c_Precily_pathways/' # params['data_url']
    # Download and unpack the data in CANDLE_DATA_DIR
    candle.file_utils.get_file(fname, origin)


def candle_main():
    params = initialize_parameters()
    preprocess(params)

if __name__ == "__main__":
    candle_main()
