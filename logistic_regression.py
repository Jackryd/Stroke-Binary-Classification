from preprocess import preprocess_data
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, roc_auc_score
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import csv


file = "Data\healthcare-dataset-stroke-data.csv"


for log in ["True", "False"]:
    for num_pca_components in [16, 14, 12, 10, 8, 6, 4]:
        X_train, X_test, Y_train, Y_test = preprocess_data(
            file, log, num_pca_components
        )
        param_grid = {
            "C": [0.1, 1, 10],
            "solver": ["liblinear", "lbfgs", "saga"],
        }
        # Create a logistic regression model
        model = LogisticRegression(max_iter=1000, random_state=0)
        # Create an instance of GridSearchCV
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
        # Fit the GridSearchCV object on the training data
        grid_search.fit(X_train, Y_train)
        # Get the best hyperparameters and best model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        # Make predictions on the test data using the best model
        Y_pred = best_model.predict(X_test)
        y_pred_prob = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(Y_test, y_pred_prob)
        roc_auc = roc_auc_score(Y_test, y_pred_prob)
        cm = confusion_matrix(Y_test, Y_pred)
        pca = num_pca_components if num_pca_components != None else 0
        cm = confusion_matrix(Y_test, Y_pred)
        print(cm)
        f1 = f1_score(Y_test, Y_pred, average="macro")
        data = [
            f1,
            log,
            pca,
            cm,
            best_model,
            best_params,
            roc_auc,
            fpr,
            tpr,
            thresholds,
        ]
        # data = [f1, fill, log, k_rf, k_chi2, sample, pca, cm]
        with open("data_logistic_regression.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(data)
