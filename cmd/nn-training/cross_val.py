"""
Runs the cross-validation experiments for the baseball-classifier.
This script is designed to be run on the DoC machines
"""

import multiprocessing
import os
import re
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from nn.model_architectures import lookup_model_by_network_type
from nn.utils import fetch_extracted_features, read_pkl_data, cross_val_iterator, train_concept_model, \
    calculate_concept_metrics

random_seed = 42
project_dir = '/vol/bitbucket/rp218/luke-for-roko'
large_data_dir = "/vol/bitbucket/rp218/Thesis_Data"
pickle_path = os.path.join(project_dir, "Extracted_Concepts/final_dict_new_codex.pkl")
FEATURE_EXTRACTOR = 'Resnet50V2'
features_dir = f'{large_data_dir}/Feature_vectors_{FEATURE_EXTRACTOR}'
model_dir = f"{large_data_dir}/Models"

num_classes = 5
classes = ['strike', 'ball', 'play', 'foul', 'out']
n_concepts = 78

class_dict = {
    'strike': 0,
    'ball': 1,
    'play': 2,
    'foul': 3,
    'out': 4
}

# network_types = ['concept_Conv_attn', 'concept_LSTM_attn']
# num_hidden = [128, 512]
# dropout_p = [0.1, 0.3]
gammas = [0.5, 1, 2]

# network_type = 'Conv1D'
# network_type = 'concept_Conv'
network_type = 'concept_Conv_attn'
# network_type = 'LSTM'
# network_type = 'concept_LSTM'
# network_type = 'concept_LSTM_attn'
# network_type = 'concept_attn'

# specifying hyper-parameters
batch_size = 16
cross_val_k = 10


def train_and_val(network_type, X_train, X_val, y_train, y_val, concept_train, concept_val, results_dict):
    _, win_len, dim = X_train.shape
    n_concepts = concept_train.shape[1]
    model = lookup_model_by_network_type(network_type, dim, win_len, num_classes, n_concepts)
    t = int(time.time())
    model, H = train_concept_model(model, X_train, y_train, concept_train,
                                   X_val, y_val, concept_val,
                                   model_dir, t, n_concepts, batch_size=batch_size, epochs=100,
                                   name=network_type, gamma=gamma)
    print(f"Results for model at time {t} with gamma {gamma}")
    cf_matrix, accuracy, macro_f1, mismatch, y_pred, cf_concepts, accuracy_concepts = calculate_concept_metrics(
        model, X_val, y_val, concept_val)
    print('Accuracy : {}'.format(accuracy))
    print('F1-score : {}'.format(macro_f1))
    print(cf_matrix)
    print(cf_concepts)
    print(accuracy_concepts)
    print()

    results_name = f"g{gamma}"
    if results_name not in results_dict:
        results_dict[results_name] = [accuracy]
    else:
        results_dict[results_name] = results_dict[results_name] + [accuracy]


if __name__ == "__main__":
    labels, explanations = read_pkl_data(pickle_path)
    labels, X = fetch_extracted_features(labels, features_dir)
    print(X.shape)
    print(labels)

    # Preprocess concept_matrix
    concept_matrix = np.stack(labels['concepts'].values, axis=0)
    idx = np.argwhere(np.all(concept_matrix[..., :] == 0, axis=0))
    concept_matrix = np.delete(concept_matrix, idx, axis=1)
    concept_matrix = concept_matrix[:, :n_concepts]
    print(concept_matrix.shape)

    y = np.array([class_dict[label] for label in labels['label']])
    y_binary = tf.keras.utils.to_categorical(y, num_classes)
    print(y_binary.shape)
    print(X.shape)

    X_trainval, X_test, y_trainval, y_test, concept_trainval, concept_test \
        = train_test_split(X, y_binary, concept_matrix, test_size=0.15, random_state=random_seed)

    manager = multiprocessing.Manager()
    results_dict = manager.dict()

    for X_train, X_val, y_train, y_val, concept_train, concept_val \
            in cross_val_iterator(X_trainval, y_trainval, concept_trainval, cross_val_k=cross_val_k):
        for gamma in gammas:
            p = multiprocessing.Process(target=train_and_val,
                                        args=(network_type, X_train, X_val, y_train, y_val, concept_train, concept_val,
                                              results_dict))
            p.start()
            p.join()

    best_sum = 0
    best_key = ""
    for k, v in results_dict.items():
        curr_sum = sum(v)
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_key = k

    match_obj = re.match(r"^g([\d.]+)", best_key)
    best_gamma = float(match_obj.groups()[0])
    print(f"The best gamma is {best_gamma}")
    print("Here is the full result dictionary:")
    print(results_dict)

    _, win_len, dim = X_trainval.shape
    n_concepts = concept_trainval.shape[1]
    model = lookup_model_by_network_type(network_type, dim, win_len, num_classes, n_concepts)

    t = int(time.time())
    model, H = train_concept_model(model, X_trainval, y_trainval, concept_trainval,
                                   X_test, y_test, concept_test,
                                   model_dir, t, n_concepts, batch_size=batch_size, epochs=100,
                                   name=network_type, gamma=best_gamma)
    print(f"Final results:")
    cf_matrix, accuracy, macro_f1, mismatch, y_pred, cf_concepts, accuracy_concepts = calculate_concept_metrics(
        model, X_test, y_test, concept_test)

    print('Accuracy : {}'.format(accuracy))
    print('F1-score : {}'.format(macro_f1))
    print(cf_matrix)
    print(cf_concepts)
    print(accuracy_concepts)
