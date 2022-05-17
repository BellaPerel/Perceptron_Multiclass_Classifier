import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd


class PerceptronClassifier:
    def __init__(self):
        """
        Constructor for the PerceptronClassifier.
        """
        self.w_array = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        This method trains a multiclass perceptron classifier on a given training set X with label set y.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
        Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
        Array datatype is guaranteed to be np.uint8.
        """
        ones_vec = np.ones(shape=(X.shape[0], 1))
        X_updated = np.append(ones_vec, X, axis=1)
        k = (np.unique(y)).size
        w_array = np.zeros(shape=(k, X_updated.shape[1]))
        not_all_labels_are_right = True
        y_prediction_max_arg_w = -1  # finding the w index which gives us the best result
        while not_all_labels_are_right:
            not_all_labels_are_right = False
            # iterate over every point in the data matrix
            for row_num in range(len(X_updated)):
                max_multiplication = float('-inf')
                # iterate over every vector w_k
                for index, w_i in enumerate(w_array):
                    multiplication = np.matmul(w_i, X_updated[row_num])
                    if multiplication > max_multiplication:
                        max_multiplication = multiplication
                        y_prediction_max_arg_w = index
                if y[row_num] != y_prediction_max_arg_w:
                    w_array[y[row_num]] += X_updated[row_num]
                    w_array[y_prediction_max_arg_w] -= X_updated[row_num]
                    not_all_labels_are_right = True
        self.w_array = w_array.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call PerceptronClassifier.fit before calling this method.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
        Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """
        y_prediction_vec = np.array([], dtype=np.uint8)
        y_prediction_max_arg_w = -1  # finding the w index which gives us the best result
        ones_vec = np.ones(shape=(X.shape[0], 1))
        X_updated = np.append(ones_vec, X, axis=1)
        # iterate over every point in the data matrix
        for row_num in range(len(X_updated)):
            max_multiplication = float('-inf')
            # iterate over every vector w_k
            for index, w_i in enumerate(self.w_array):
                multiplication = np.matmul(w_i, X_updated[row_num])
                if multiplication > max_multiplication:
                    max_multiplication = multiplication
                    y_prediction_max_arg_w = index
            y_prediction_vec = np.append(y_prediction_vec, y_prediction_max_arg_w)
        return y_prediction_vec


if __name__ == "__main__":

    print("*" * 20)
    print("Started HW2_ID1_ID2.py")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}")

    print("Initiating PerceptronClassifier")
    model = PerceptronClassifier()
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    is_separable = model.fit(X, y)
    print("Done")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y.ravel()) / y.shape[0]
    print(f"Train accuracy: {accuracy * 100 :.2f}%")

    print("*" * 20)
