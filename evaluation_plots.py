import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

f1_scores = {}


def parse_cm_string(cm_string):
    # Remove whitespace and split into rows
    rows = cm_string.strip().split("\n")
    # Remove brackets and split each row into numbers
    data = [[int(num) for num in re.findall(r"\d+", row)] for row in rows]
    return np.array(data)


def plot_best_f1(file, name):
    # f1,log,pca,cm,colsamle_bytree,gamma,learning_rate,max_depth,n_estimators,reg_alpha,reg_lambda,roc_auc,fpr,tpr,thresholds,
    print(name)
    df = pd.read_csv(file)
    df = df[["f1", "cm", "roc_auc"]]

    df = df.sort_values("f1", axis=0, ascending=False)
    cm = df["cm"][:1].values[0]
    cm = parse_cm_string(cm)

    f1_scores[name] = df["f1"][:1].values[0]

    print(type(cm))

    # Create a figure and axis
    plt.figure(figsize=(10, 8))

    # Plot the confusion matrix using seaborn
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    # Set labels and title
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # Save the plot
    plt.savefig(f"plots/confusion_matrix_{name}.png")
    # plt.show()


def compare_f1_scores(model_performances):
    # Create lists of models and their corresponding performances
    models = list(model_performances.keys())
    performances = list(model_performances.values())

    # Create the bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, performances)

    # Customize the plot
    plt.title("Model Performance Comparison", fontsize=16)
    plt.xlabel("Models", fontsize=12)
    plt.ylabel("Performance Score", fontsize=12)
    plt.ylim(0.0, 1.0)  # Adjust y-axis to focus on the range of scores

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
        )

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    # Save the plot
    plt.savefig("plots/model_comparison.png")

    # Display the plot (optional)
    # plt.show()


names = [
    "Random Forest",
    "Neural Network",
    "XGBoost",
    "Naïve Bayes",
    "Logistic Regression",
]

for file, name in zip(
    [
        "data_random_forest",
        "data_neural_network",
        "data_xgboost_classifier",
        "BayesCat",
        "data_logistic_regression",
    ],
    names,
):
    plot_best_f1(f"C:\Code\Stroke\{file}.csv", name)


compare_f1_scores(f1_scores)
