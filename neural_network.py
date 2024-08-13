from preprocess import preprocess_data
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, roc_auc_score
import os
import csv


def create_model(
    X_train,
    num_layers=2,
    layer_units=256,
    dropout_rate=0.5,
    l1_reg=0.001,
    l2_reg=0.001,
    activation="relu",
    opt="adam",
    learning_rate=0.01,
):
    model = keras.Sequential()
    model.add(layers.BatchNormalization(input_shape=(X_train.shape[1],)))
    for _ in range(num_layers):
        model.add(layers.Dense(layer_units, activation=activation))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
    model.add(
        layers.Dense(
            1,
            activation="sigmoid",
            kernel_regularizer=keras.regularizers.l1_l2(l1_reg, l2_reg),
        )
    )
    if opt == "adam":
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        opt = keras.optimizers.SGD()

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )
    return model


def predict_with_bn(model, X, batch_size=32):
    # Function to perform prediction while maintaining BatchNormalization in inference mode
    predictions = []
    num_samples = len(X)
    for i in range(0, num_samples, batch_size):
        batch = X[i : i + batch_size]
        batch_predictions = model(batch, training=False)
        predictions.extend(batch_predictions.numpy())
    return np.array(predictions)


def find_best_model(file, save_folder="best_models"):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    layer_options = [1, 2, 3]  # Number of dense layers to try
    units_options = [32, 64, 128]  # Number of units in each dense layer to try
    dropout_rate_options = [0.2, 0.5, 0.8]
    l1_reg_options = [0.001, 0.01]
    l2_reg_options = [0.001, 0.01]
    activation_options = ["elu", "relu"]
    learning_rate_options = [0.001, 0.001]
    batch_size_options = [32, 64]
    patience_options = [8, 11]
    optimizers = ["sgd", "adam"]
    fill = "median"
    pca_options = [4, 6, 8, 10, 12, 14, 16, 18]

    for activation in activation_options:
        for l1_reg in l1_reg_options:
            for l2_reg in l2_reg_options:
                for learning_rate in learning_rate_options:
                    for opt in optimizers:
                        for patience in patience_options:
                            for batch_size in batch_size_options:
                                for pca in pca_options:
                                    for dropout_rate in dropout_rate_options:
                                        for num_layers in layer_options:
                                            for layer_units in units_options:
                                                for log in [True, False]:
                                                    (
                                                        X_train,
                                                        X_test,
                                                        Y_train,
                                                        Y_test,
                                                    ) = preprocess_data(
                                                        file,
                                                        log,
                                                        num_pca_components=pca,
                                                    )
                                                    model = create_model(
                                                        X_train,
                                                        num_layers=num_layers,
                                                        layer_units=layer_units,
                                                        dropout_rate=dropout_rate,
                                                        l1_reg=l1_reg,
                                                        l2_reg=l2_reg,
                                                        activation=activation,
                                                        opt=opt,
                                                        learning_rate=learning_rate,
                                                    )
                                                    early_stopping = (
                                                        keras.callbacks.EarlyStopping(
                                                            patience=patience,
                                                            min_delta=0.01,
                                                            restore_best_weights=True,
                                                        )
                                                    )
                                                    checkpoint_filepath = os.path.join(
                                                        save_folder,
                                                        f"model_{num_layers}_{layer_units}_{dropout_rate}_{l1_reg}_{l2_reg}_{activation}.h5",
                                                    )
                                                    model_checkpoint = (
                                                        keras.callbacks.ModelCheckpoint(
                                                            checkpoint_filepath,
                                                            monitor="val_loss",
                                                            save_best_only=True,
                                                        )
                                                    )
                                                    callbacks = [
                                                        early_stopping,
                                                        model_checkpoint,
                                                    ]
                                                    history = model.fit(
                                                        X_train,
                                                        Y_train,
                                                        validation_split=0.2,
                                                        batch_size=batch_size,
                                                        epochs=200,
                                                        callbacks=callbacks,
                                                        verbose=0,
                                                    )
                                                    best_score = f1_score(
                                                        Y_train,
                                                        (
                                                            predict_with_bn(
                                                                model, X_train
                                                            )
                                                            >= 0.5
                                                        ).astype(int),
                                                    )
                                                    best_params = {
                                                        "num_layers": num_layers,
                                                        "layer_units": layer_units,
                                                        "dropout_rate": dropout_rate,
                                                        "l1_reg": l1_reg,
                                                        "l2_reg": l2_reg,
                                                        "activation": activation,
                                                        "learning_rate": learning_rate,
                                                        "batch_size": batch_size,
                                                        "patience": patience,
                                                    }
                                                    # Load and evaluate best model from the saved checkpoint
                                                    model = keras.models.load_model(
                                                        checkpoint_filepath
                                                    )
                                                    Y_pred_prob = predict_with_bn(
                                                        model, X_test
                                                    )  # Use custom predict function

                                                    fpr, tpr, thresholds = roc_curve(
                                                        Y_test, Y_pred_prob
                                                    )
                                                    roc_auc = roc_auc_score(
                                                        Y_test, Y_pred_prob
                                                    )

                                                    Y_pred = (
                                                        Y_pred_prob >= 0.5
                                                    ).astype(int)
                                                    cm = confusion_matrix(
                                                        Y_test, Y_pred
                                                    )
                                                    print(cm)
                                                    f1 = f1_score(
                                                        Y_test, Y_pred, average="macro"
                                                    )
                                                    data = [
                                                        f1,
                                                        num_layers,
                                                        layer_units,
                                                        dropout_rate,
                                                        pca,
                                                        l1_reg,
                                                        l2_reg,
                                                        activation,
                                                        learning_rate,
                                                        batch_size,
                                                        patience,
                                                        opt,
                                                        fill,
                                                        log,
                                                        cm,
                                                        roc_auc,
                                                        fpr,
                                                        tpr,
                                                        thresholds,
                                                    ]
                                                    with open(
                                                        "C:\Code\Stroke\data_neural_network.csv",
                                                        "a",
                                                    ) as f:
                                                        writer = csv.writer(f)
                                                        writer.writerow(data)

    return best_params, best_score


file = "Data\healthcare-dataset-stroke-data.csv"
best_params, best_score = find_best_model(file)
