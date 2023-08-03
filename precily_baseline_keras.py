import argparse
import json
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau

import candle

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def model_builder(params):
    model = keras.Sequential()
    model.add(
        keras.layers.Dense(
            params["input_dim"],
            input_dim=params["input_dim"],
            activation=params["activation"],
        )
    )
    model.add(keras.layers.Dense(units=512, activation=params["activation"]))
    model.add(keras.layers.Dropout(params["dropout"]))

    #  the length of the dense list == layers and the values == units
    for layer in range(len(params["dense"])):
        model.add(keras.layers.Dense(units=layer, activation=params["activation"]))
        model.add(keras.layers.Dropout(params["dropout"]))
        model.add(keras.layers.Dense(1))

    kerasDefaults = candle.keras_default_config()
    optimizer = candle.build_optimizer(
        params["optimizer"], params["learning_rate"], kerasDefaults
    )
    model.compile(optimizer=optimizer, loss=params["loss"])
    return model


def main(params):
    train_data = params["train_data"]  # train_set

    output_dir = params["output_dir"]
    model_name = params["model_name"]

    train_set = pd.read_csv(train_data)

    CL_x = train_set[train_set.columns[0]].unique()
    CL_x = list(CL_x)

    CL_x = train_set[train_set.columns[0]].unique()
    CL_x = list(CL_x)

    ##Perform hyperparameter tuning
    for i in range(1):
        A = set(CL_x[: i * len(CL_x) // 5] + CL_x[(i + 1) * len(CL_x) // 5 :])
        B = set(CL_x[i * len(CL_x) // 5 : (i + 1) * len(CL_x) // 5])

        train, val = [], []
        for j in train_set.to_numpy():
            if j[0] in A:
                train.append(j)
            else:
                val.append(j)

        train = pd.DataFrame(train)
        val = pd.DataFrame(val)
        # train.to_csv(path+"Train_Set_"+str(i+1)+".csv",index=False)
        # val.to_csv(path+"Validation_Set_"+str(i+1)+".csv",index=False)
        X_train = train.iloc[:, 2:-1]
        Y_train = train.iloc[:, -1:]
        X_val = val.iloc[:, 2:-1]
        Y_val = val.iloc[:, -1:]
        train, test = None, None
        X_train.to_csv("Train_Set_" + str(i + 1) + ".csv", index=False)
        Y_train.to_csv("Train_Set_truth_" + str(i + 1) + ".csv", index=False)
        X_val.to_csv("Val_Set_" + str(i + 1) + ".csv", index=False)
        Y_val.to_csv("Val_Set_truth_" + str(i + 1) + ".csv", index=False)

        stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

        # Build the model with the optimal hyperparameters
        model = model_builder(params)
        
        ckpt = candle.CandleCkptKeras(params, verbose=False)
        ckpt.set_model(model)
        J = ckpt.restart(model)
        if J is not None:
            initial_epoch = J["epoch"]
            print("restarting from ckpt: initial_epoch: %i" % initial_epoch)
        
        output_dir = params["output_dir"]

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # calculate trainable and non-trainable params
        params.update(candle.compute_trainable_params(model))
        
        model_name = params["model_name"]
        # path = '{}/{}.autosave.model.h5'.format(output_dir, model_name)
        # checkpointer = ModelCheckpoint(filepath=path, verbose=1, save_weights_only=False, save_best_only=True)
        csv_logger = CSVLogger("{}/training.log".format(output_dir))
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=10,
            verbose=1,
            mode="auto",
            min_delta=0.0001,
            cooldown=0,
            min_lr=0,
        )
        candleRemoteMonitor = candle.CandleRemoteMonitor(params=params)
        timeoutMonitor = candle.TerminateOnTimeOut(params["timeout"])
        
        history = model.fit(
            X_train,
            Y_train,
            epochs=params['epochs'],
            verbose=1,
            batch_size=params['batch_size'],
            validation_data=(X_val, Y_val),
            callbacks=[csv_logger, reduce_lr, candleRemoteMonitor, timeoutMonitor, ckpt]
        )
        
        
        
        model.save("precily_cv" + ".hdf5")
        model = None
        
    return history

if __name__ == '__main__':
    main(params)
