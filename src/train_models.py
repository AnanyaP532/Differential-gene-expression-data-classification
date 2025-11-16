#Train multiple classifiers on processed data and save evaluation metrics
#python src/train_models.py --input results/processed.csv --output results/metrics.json --figdir results/figures

import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def load_data(path):
    df = pd.read_csv(path)
    if 'target' not in df.columns:
        raise ValueError('Processed CSV must contain a "target" column.')
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y

def train_and_evaluate(X_train, X_test, y_train, y_test, models):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            'accuracy': float(accuracy_score(y_test, preds)),
            'precision': float(precision_score(y_test, preds, zero_division=0)),
            'recall': float(recall_score(y_test, preds, zero_division=0)),
            'f1': float(f1_score(y_test, preds, zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test, preds).tolist()
        }
    return results

def plot_comparison(results, outpath):
    df = pd.DataFrame(results).T
    df = df[['accuracy','precision','recall','f1']]
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.plot.bar(rot=45)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def main(infile, outfile, figdir):
    X, y = load_data(infile)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=32)
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100),
        'AdaBoost': AdaBoostClassifier(),
        'SVC': SVC(probability=False),
        'GaussianNB': GaussianNB()
    }
    results = train_and_evaluate(X_train, X_test, y_train, y_test, models)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {outfile}")
 
    figpath = os.path.join(figdir, 'model_comparison.png')
    plot_comparison(results, figpath)
    print(f"Saved figure to {figpath}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train models on processed data')
    parser.add_argument('--input', required=True, help='Processed CSV input')
    parser.add_argument('--output', required=True, help='Metrics JSON output')
    parser.add_argument('--figdir', required=True, help='Directory to save figures')
    args = parser.parse_args()
    main(args.input, args.output, args.figdir)
