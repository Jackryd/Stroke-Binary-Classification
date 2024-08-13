from preprocess import preprocess_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, roc_auc_score
import csv

file = "Data\healthcare-dataset-stroke-data.csv"

for log in ["True", "False"]:
    for max_depth in [10, 15, None]:
        for min_samples_split in [2, 5, 10]:
            for max_features in ["sqrt", "log2", 0.8, 0.5]:
                for bootstrap in [True, False]:
                    for random_state in [123, 42]:
                        for n_estimators in [100, 200, 300]:
                            for num_pca_components in [18, 15, 12, 9, 6, 4]:
                                for sample in ["SMOTE", "not"]:
                                    (
                                        X_train,
                                        X_test,
                                        Y_train,
                                        Y_test,
                                    ) = preprocess_data(
                                        file,
                                        log,
                                        num_pca_components,
                                    )
                                    rf_classifier = RandomForestClassifier(
                                        n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=1,
                                        bootstrap=bootstrap,
                                        random_state=random_state,
                                        max_features=max_features,
                                    )
                                    # Train the model on the training data
                                    rf_classifier.fit(X_train, Y_train)
                                    # Predict on the test data
                                    Y_pred = rf_classifier.predict(X_test)
                                    y_pred_prob = rf_classifier.predict_proba(X_test)[
                                        :, 1
                                    ]
                                    fpr, tpr, thresholds = roc_curve(
                                        Y_test, y_pred_prob
                                    )
                                    roc_auc = roc_auc_score(Y_test, y_pred_prob)
                                    pca = (
                                        num_pca_components
                                        if num_pca_components != None
                                        else 0
                                    )
                                    cm = confusion_matrix(Y_test, Y_pred)
                                    print(cm)
                                    f1 = f1_score(Y_test, Y_pred, average="macro")
                                    data = [
                                        f1,
                                        log,
                                        pca,
                                        cm,
                                        n_estimators,
                                        max_depth,
                                        min_samples_split,
                                        max_features,
                                        bootstrap,
                                        random_state,
                                        roc_auc,
                                        fpr,
                                        tpr,
                                        thresholds,
                                    ]
                                    with open("data_random_forest.csv", "a") as f:
                                        writer = csv.writer(f)
                                        writer.writerow(data)
