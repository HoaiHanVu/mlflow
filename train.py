import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, f1_score
import xgboost as xgb
import mlflow
import mlflow.sklearn

def eval_metrics(y_true, y_pred):
    accuracy = round(accuracy_score(y_true, y_pred), 3)
    precision = round(precision_score(y_true, y_pred), 3)
    f1 = round(f1_score(y_true, y_pred), 3)
    auc = round(roc_auc_score(y_true, y_pred), 3)

    return accuracy, precision, f1, auc

if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators")
    parser.add_argument("--learning_rate")
    args = parser.parse_args()

    df = pd.read_csv("Marketing-Customer-Value-Analysis.csv")

    features_collected = ['Customer Lifetime Value', 
                          'Income', 
                          'Monthly Premium Auto',
                          'Months Since Policy Inception', 
                          'Total Claim Amount']
    label = 'Response'

    df['Response'] = df['Response'].map(lambda x: 1 if x == "Yes" else 0)

    X = df[features_collected]
    y = df['Response']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    n_estimators = int(args.n_estimators)
    learning_rate = float(args.learning_rate)

    with mlflow.start_run():
        xgb_clf = xgb.XGBClassifier(random_state = 42, 
                                  learning_rate=learning_rate, 
                                  n_estimators=n_estimators)
        
        xgb_model = xgb_clf.fit(X_train, y_train)

        yhat = xgb_model.predict(X_test)

        (accuracy, precision, f1, auc) = eval_metrics(y_test, yhat)
        
        print("XGBClassifier model (learning_rate={}, n_estimators={})".format(learning_rate, n_estimators))
        print("Accuracy: {}".format(accuracy))
        print("Precision: {}".format(precision))
        print("F1 score: {}".format(f1))
        print("AUC: {}".format(auc))

        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("n_estimators", n_estimators)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("auc", auc)

        mlflow.sklearn.log_model(xgb_model, "XGBModel")







