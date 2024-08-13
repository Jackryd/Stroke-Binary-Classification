import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTENC, SMOTEN
from imblearn.under_sampling import TomekLinks


def encode_smoking_status(status):
    if status == "never smoked":
        return 0
    elif status == "formerly smoked":
        return 1
    elif status == "currently smokes":
        return 2
    else:
        return 0.5  # For 'unknown' status


def compute_healthscore(df):
    # Make a copy of the DataFrame to avoid modifying the original data
    df_processed = df.copy()

    # Define weights for each health factor (you can adjust these weights based on relevance)
    age_weight = 0.2
    bmi_weight = 0.2
    glucose_weight = 0.2
    smoking_weight = 0.2
    hypertension_weight = 0.2

    # Encode smoking status
    df_processed["smoking_status_encoded"] = df_processed["smoking_status"].apply(
        encode_smoking_status
    )

    # Calculate the health score for each person
    df_processed["health_score"] = (
        df_processed["age"] * age_weight
        + df_processed["bmi"] * bmi_weight
        + df_processed["avg_glucose_level"] * glucose_weight
        + df_processed["smoking_status_encoded"] * smoking_weight
        + df_processed["hypertension"] * hypertension_weight
    )

    # Health score is normalized to a [0, 1] scale
    scaler = MinMaxScaler()
    df_processed["health_score"] = scaler.fit_transform(
        df_processed["health_score"].values.reshape(-1, 1)
    )

    return df_processed


def preprocess_data(file, log, num_pca_components=None, naive_bayes=False):
    df = pd.read_csv(file)
    df = df.sample(frac=1)
    df = df.drop("id", axis=1)

    df["bmi"].fillna(value=df["bmi"].median(), inplace=True)

    df = df[df["gender"] != "Other"]
    df = df[df["work_type"] != "Never_worked"]

    age_bins = [0, 15, 30, 45, 60, 75, 500]
    bmi_bins = [0, 18.5, 24.9, 29.9, 34.9, 1000]
    glucose_bins = [0, 70, 100, 125, 2000]

    df["age_category"] = pd.cut(
        df["age"],
        bins=age_bins,
        labels=["0-15", "15-30", "30-45", "45-60", "60-75", "75+"],
    )
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=bmi_bins,
        labels=["Underweight", "Normal", "Overweight", "Obese", "Extremely Obese"],
    )
    df["glucose_category"] = pd.cut(
        df["avg_glucose_level"],
        bins=glucose_bins,
        labels=["Low", "Normal", "Pre-diabetes", "Diabetes"],
    )

    # df = compute_healthscore(df)

    label_encoder = LabelEncoder()
    categorical_nominal = ["work_type", "ever_married", "gender", "Residence_type"]
    categorical_ordinal = [
        "bmi_category",
        "age_category",
        "glucose_category",
        "smoking_status",
    ]

    for column in categorical_ordinal:
        df[column] = label_encoder.fit_transform(df[column])
    df = pd.get_dummies(df, columns=categorical_nominal)

    if naive_bayes:
        df = df.drop(["age", "avg_glucose_level", "bmi"], axis=1)

    x = np.asarray(df.drop("stroke", axis=1)).astype("float32")
    y = np.asarray(df["stroke"]).astype("float32")
    X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()

    age = 0
    glucose = 3
    bmi = 4

    # Since naive bayes only takes categorical values we need to preprocess the data differently
    if naive_bayes != True:
        if log:
            X_train[:, [bmi, glucose]] = np.log(X_train[:, [bmi, glucose]])
            X_test[:, [bmi, glucose]] = np.log(X_test[:, [bmi, glucose]])

        X_train[:, [age]] = standard_scaler.fit_transform(X_train[:, [age]])
        X_train[:, [glucose, bmi]] = minmax_scaler.fit_transform(
            X_train[:, [glucose, bmi]]
        )
        X_test[:, [age]] = standard_scaler.transform(X_test[:, [age]])
        X_test[:, [glucose, bmi]] = minmax_scaler.transform(X_test[:, [glucose, bmi]])

        categorical = [
            i for i in range(len(X_train[0])) if i not in [age, glucose, bmi]
        ]

        tl = TomekLinks()
        X_train, Y_train = tl.fit_resample(X_train, Y_train)
        X_test, Y_test = tl.fit_resample(X_test, Y_test)

        sm = SMOTENC(categorical_features=categorical, random_state=42)
    else:
        sm = SMOTEN(random_state=42)
    X_test, Y_test = sm.fit_resample(X_test, Y_test)
    X_train, Y_train = sm.fit_resample(X_train, Y_train)

    if not naive_bayes:
        if num_pca_components is not None:
            pca = PCA(n_components=num_pca_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

    return X_train, X_test, Y_train, Y_test
