from preprocess import preprocess_data
import csv
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, confusion_matrix


file = "Data\healthcare-dataset-stroke-data.csv"

for log in [True, False]:
    for num_pca_components in [16, 12, 9, 6, 5]:
        X_train, X_test, Y_train, Y_test = preprocess_data(
            file, log, num_pca_components, naive_bayes=True
        )
        model = CategoricalNB()
        model.fit(X_train, Y_train)

        Y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(Y_test, y_pred_prob)
        roc_auc = roc_auc_score(Y_test, y_pred_prob)
        pca = num_pca_components if num_pca_components != None else 0
        cm = confusion_matrix(Y_test, Y_pred)
        print(cm)
        f1 = f1_score(Y_test, Y_pred, average="macro")
        # data = [f1, fill, log, k_rf, k_chi2, sample, pca, cm, best_model, best_params]
        data = [f1, log, pca, cm, roc_auc, fpr, tpr, thresholds]
        # data = [f1, fill, log, k_rf, k_chi2, sample, pca, cm]
        with open("BayesCat.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(data)
