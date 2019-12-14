import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix


# Reding the Test.csv file to test and find the accuracy and confusion metrics
def test_prediction(column_to_predict):
    df = pd.read_csv('data/test.csv', sep=chr(1), error_bad_lines=False)

    df = df[df[column_to_predict].notna() & (df['title'] + df['description']).notna()]  # selecting only non null gender
    test_labels = df[column_to_predict]  # Column to predict
    test_data = df['title'] + df['description']  # Text columns
    test_data = test_data.apply(lambda x: np.str_(x))

    file_name = 'outputs' + column_to_predict + ".model"
    with open(file_name, 'rb') as file:
        pickle_model = pickle.load(file)

    predicted = pickle_model.predict(test_data)

    print(metrics.confusion_matrix(test_labels, predicted))
    print(metrics.accuracy_score(test_labels, predicted))
    print(metrics.classification_report(test_labels, predicted))

    mat = confusion_matrix(test_labels, predicted)
    plt.figure(figsize=(4, 4))
    sns.set()
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=np.unique(test_labels),
                yticklabels=np.unique(test_labels))
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    # Save confusion matrix to results folder
    plt.savefig('results/' + column_to_predict + 'confusion_matrix.png')
