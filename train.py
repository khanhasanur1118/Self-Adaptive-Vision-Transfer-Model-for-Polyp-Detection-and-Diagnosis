
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from glob import glob
from data import load_data, tf_dataset
from model import build_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.metrics import Recall, Precision

# -------- Custom Metrics --------

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        y_true = y_true.astype(np.float32)
        y_pred = y_pred.astype(np.float32)
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred) - intersection
        return np.float32((intersection + 1e-15) / (union + 1e-15))
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def dice_coef(y_true, y_pred):
    def f(y_true, y_pred):
        y_true = y_true.astype(np.float32)
        y_pred = y_pred.astype(np.float32)
        intersection = np.sum(y_true * y_pred)
        return np.float32((2. * intersection + 1e-15) / (np.sum(y_true) + np.sum(y_pred) + 1e-15))
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def mcc(y_true, y_pred):
    def f(y_true, y_pred):
        y_true = y_true.astype(np.bool_)
        y_pred = (y_pred > 0.5).astype(np.bool_)
        TP = np.sum(y_true & y_pred)
        TN = np.sum(~y_true & ~y_pred)
        FP = np.sum(~y_true & y_pred)
        FN = np.sum(y_true & ~y_pred)
        numerator = (TP * TN) - (FP * FN)
        denominator = np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)) + 1e-15
        return np.float32(numerator / denominator)
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def fps(y_true, y_pred):
    def f(y_true, y_pred):
        y_true = y_true.astype(np.bool_)
        y_pred = (y_pred > 0.5).astype(np.bool_)
        FP = np.sum(~y_true & y_pred)
        N = np.sum(~y_true)
        return np.float32((FP + 1e-15) / (N + 1e-15))
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

# -------- Training Pipeline --------

if __name__ == "__main__":
    path = "cvc dataset"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)

    ## Hyperparameters
    batch = 8
    lr = 1e-4
    epochs = 20

    train_dataset = tf_dataset(train_x, train_y, batch=batch)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)

    model = build_model()
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    metrics = [
        "accuracy",
        Recall(name="recall"),
        Precision(name="precision"),
        iou,
        dice_coef,
        mcc,
        fps
    ]

    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=metrics)

    callbacks = [
        ModelCheckpoint("files/model.h5", save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
        CSVLogger("files/data.csv"),
        TensorBoard(log_dir="logs"),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    train_steps = len(train_x) // batch + int(len(train_x) % batch != 0)
    valid_steps = len(valid_x) // batch + int(len(valid_x) % batch != 0)

    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks
    )





