import candle
import os
from precily_baseline_keras import main
import json

# This should be set outside as a user environment variable
os.environ['CANDLE_DATA_DIR'] = '/homes/ac.tfeng/git/Precily'
file_path = os.path.dirname(os.path.realpath(__file__))

required=None
additional_definitions=None

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
    params['train_data'] = os.environ['CANDLE_DATA_DIR'] + 'Fig1/Fig1c/Fig1c_Precily_pathways/' + params['train_data']
    """ 
    params["train_data"] = candle.get_file(params['train_data'], origin, datadir=params['data_url'], cache_subdir=None)
    """
    return params

def run(params):
    params['data_type'] = str(params['data_type'])
    with open ((params['output_dir']+'/params.json'), 'w') as outfile:
        json.dump(params, outfile)
    scores = main(params)
    with open(params['output_dir'] + "/scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    #print('IMPROVE_RESULT RMSE:\t' + str(scores['rmse']))
    print("\nIMPROVE_RESULT val_loss:\t{}\n".format(scores["val_loss"]))

def candle_main():
    params = initialize_parameters()
    params = preprocess(params)
    run(params)

if __name__ == "__main__":
    candle_main()
