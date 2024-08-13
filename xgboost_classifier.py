from preprocess import preprocess_data
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, roc_auc_score
import csv

file = "Data\healthcare-dataset-stroke-data.csv"


for log in [True, False]:
    for num_pca_components in [18, 13, 8, 4]:
        for colsamle_bytree in [0.2, 0.5, 0.8]:
            for gamma in [0.1, 0.3, 0.5]:
                for learning_rate in [0.1, 0.01, 0.001]:
                    for max_depth in [5, 7, 10]:
                        for n_estimators in [100, 200, 300]:
                            for reg_alpha in [0, 0.01, 0.1]:
                                for reg_lambda in [0, 0.01, 0.1]:
                                    X_train, X_test, Y_train, Y_test = preprocess_data(
                                        file, log, num_pca_components
                                    )
                                    model = XGBClassifier(
                                        colsample_bytree=colsamle_bytree,
                                        gamma=gamma,
                                        learning_rate=learning_rate,
                                        max_depth=max_depth,
                                        min_child_weight=5,
                                        n_estimators=n_estimators,
                                        subsample=0.5,
                                        reg_alpha=reg_alpha,
                                        reg_lambda=reg_lambda,
                                    )
                                    model.fit(X_train, Y_train)
                                    Y_pred = model.predict(X_test)

                                    Y_pred = model.predict(X_test)
                                    y_pred_prob = model.predict_proba(X_test)[:, 1]
                                    fpr, tpr, thresholds = roc_curve(
                                        Y_test, y_pred_prob
                                    )
                                    roc_auc = roc_auc_score(Y_test, y_pred_prob)

                                    cm = confusion_matrix(Y_test, Y_pred)
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
                                        colsamle_bytree,
                                        gamma,
                                        learning_rate,
                                        max_depth,
                                        n_estimators,
                                        reg_alpha,
                                        reg_lambda,
                                        roc_auc,
                                        fpr,
                                        tpr,
                                        thresholds,
                                    ]
                                    with open("data_xgboost_classifier.csv", "a") as f:
                                        writer = csv.writer(f)
                                        writer.writerow(data)
