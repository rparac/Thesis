from typing import Dict, Any, List, Iterator, Tuple

import numpy as np
import os

import pandas as pd
import regex
import tensorflow as tf
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Model
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


def train_model(model, X_train, y_train, X_val, y_val, model_dir, t, name, batch_size=32, epochs=50):
    # Only 1 dimension
    is_binary = len(y_train.shape) == 1
    metric = 'binary_accuracy' if is_binary else 'categorical_accuracy'
    loss = binary_crossentropy if is_binary else categorical_crossentropy

    model.compile(loss=loss,
                  optimizer='adam',
                  metrics=[metric])

    # checkpoint
    chk_path = os.path.join(model_dir, 'best_{}_{}.h5'.format(name, t))
    checkpoint = ModelCheckpoint(chk_path, monitor=f'val_{metric}', verbose=1, save_best_only=True,
                                 mode='max')
    tensorboard = TensorBoard(log_dir="logs/{}_{}".format(name, t))
    callbacks_list = [checkpoint, tensorboard]

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        shuffle=True,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks_list)

    # Saving the model
    model.save(os.path.join(model_dir, 'final_{}_{}.h5'.format(name, t)))

    # Loading best trained model
    model = load_model(chk_path)

    return model, history


# Train concept model using the joint method
def train_concept_model(model, X_train, y_train, c_train, X_val, y_val, c_val,
                        model_dir, t, n_concepts, name, batch_size=32, epochs=50, gamma=1):
    # Only 1 dimension
    is_binary = len(y_train.shape) == 1
    metric = 'binary_accuracy' if is_binary else 'categorical_accuracy'
    final_loss = binary_crossentropy if is_binary else categorical_crossentropy

    losses = {
        "c_probs": binary_crossentropy,
        "probs": final_loss,
    }

    model.compile(loss=losses,
                  loss_weights=[gamma, 1],
                  optimizer='adam',
                  metrics=[metric])

    # checkpoint
    chk_path = os.path.join(model_dir, 'best_{}_{}_{}.h5'.format(name, n_concepts, t))
    checkpoint = ModelCheckpoint(chk_path, monitor=f'val_probs_{metric}',
                                 verbose=1, save_best_only=True, mode='max')
    tensorboard = TensorBoard(log_dir="logs/{}_{}".format(name, t))
    callbacks_list = [checkpoint, tensorboard]

    history = model.fit(X_train, {'probs': y_train, 'c_probs': c_train},
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        shuffle=True,
                        validation_data=(X_val, {'probs': y_val, 'c_probs': c_val}),
                        callbacks=callbacks_list)

    # Saving the model
    model.save(os.path.join(model_dir, 'final_{}_{}_{}.h5'.format(name, n_concepts, t)))
    # Loading the best model
    model = load_model(chk_path)
    return model, history


# Train model using the sequential training approach - freeze the concept prediction before training
# the final label
def train_concept_model_sequential(model_tuple, X_train, y_train, c_train, X_val, y_val, c_val, model_dir, t,
                                   n_concepts, name, epochs=50, batch_size=32):
    concept_model, full_model = model_tuple

    concept_model.compile(loss=binary_crossentropy,
                          optimizer='adam',
                          metrics=['accuracy'])

    # checkpoint
    chk_path = os.path.join(model_dir, 'best_c_{}_{}.h5'.format(name, t))
    checkpoint = ModelCheckpoint(chk_path, monitor='val_accuracy', verbose=1, save_best_only=True,
                                 save_weights_only=True,
                                 mode='max')
    tensorboard = TensorBoard(log_dir="logs/c_{}_{}".format(name, t))
    callbacks_list = [checkpoint, tensorboard]

    concept_model.fit(X_train, c_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      shuffle=True,
                      validation_data=(X_val, c_val),
                      callbacks=callbacks_list)

    # Loading best trained model
    concept_model.load_weights(chk_path)

    # Only 1 dimension
    is_binary = len(y_train.shape) == 1

    final_loss = binary_crossentropy if is_binary else categorical_crossentropy
    metrics = ['binary_accuracy'] if is_binary else ['categorical_accuracy']
    concept_model.trainable = False
    full_model.compile(loss=[binary_crossentropy, final_loss],
                       loss_weights=[0, 1],
                       optimizer='adam',
                       metrics=metrics)

    # checkpoint
    chk_path = os.path.join(model_dir, 'best_{}_{}_{}.h5'.format(name, n_concepts, t))
    monitor_metric = "val_probs_binary_accuracy" if is_binary else "val_probs_categorical_accuracy"
    checkpoint = ModelCheckpoint(chk_path, monitor=monitor_metric,
                                 verbose=1, save_best_only=True, mode='max')
    tensorboard = TensorBoard(log_dir="logs/{}_{}".format(name, t))
    callbacks_list = [checkpoint, tensorboard]

    # full_model.fit(train_ds, epochs=epochs, batch_size=batch_size, validation_data=val_ds)
    history = full_model.fit(X_train, {'probs': y_train, 'c_probs': c_train},
                             batch_size=batch_size,
                             epochs=epochs,
                             verbose=1,
                             shuffle=True,
                             validation_data=(X_val, {'probs': y_val, 'c_probs': c_val}),
                             callbacks=callbacks_list)

    # Saving the model
    full_model.save(os.path.join(model_dir, 'final_{}_{}_{}.h5'.format(name, n_concepts, t)))
    # Loading the best model
    full_model = load_model(chk_path)

    return full_model, history


def calculate_metrics_binary(model, X_test, y_true):
    y_pred = model.predict(X_test).squeeze()
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    mismatch = np.where(y_true != y_pred)
    cf_matrix = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    # micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    return cf_matrix, accuracy, macro_f1, mismatch, y_pred


def calculate_metrics(model, X_test, y_test_binary):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test_binary, axis=1)
    mismatch = np.where(y_true != y_pred)
    cf_matrix = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    # micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    return cf_matrix, accuracy, macro_f1, mismatch, y_pred


def calculate_concept_metrics_binary(model, X_test, y_true, c_test):
    c_pred, y_pred = model.predict(X_test)

    y_pred = y_pred.squeeze()
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    mismatch = np.where(y_true != y_pred)
    cf_matrix = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    # micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    c_test = c_test.flatten()
    c_pred = c_pred.flatten()
    c_pred[c_pred <= 0.5] = 0
    c_pred[c_pred > 0.5] = 1
    cf_concepts = confusion_matrix(c_test, c_pred)
    accuracy_concepts = accuracy_score(c_test, c_pred)

    return cf_matrix, accuracy, macro_f1, mismatch, y_pred, cf_concepts, accuracy_concepts


def calculate_concept_metrics(model, X_test, y_test_binary, c_test):
    pred = model.predict(X_test)
    y_pred = np.argmax(pred[1], axis=1)
    y_true = np.argmax(y_test_binary, axis=1)
    mismatch = np.where(y_true != y_pred)
    cf_matrix = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    # micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    c_test = c_test.flatten()
    c_pred = pred[0]
    c_pred = c_pred.flatten()
    c_pred[c_pred <= 0.5] = 0
    c_pred[c_pred > 0.5] = 1
    cf_concepts = confusion_matrix(c_test, c_pred)
    accuracy_concepts = accuracy_score(c_test, c_pred)

    return cf_matrix, accuracy, macro_f1, mismatch, y_pred, cf_concepts, accuracy_concepts


# Get output after attention
def get_attention(model, layer_name, input_data):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(input_data)

    return intermediate_output


# Visualises the concepts after attn with a bar chart
def visualize_concepts(test_input, model, concepts_text: np.ndarray, inv_class_dict, save_name=None, verbose=False):
    test_input = np.expand_dims(test_input, axis=0)
    pred = model.predict(test_input)
    pred_label = np.argmax(pred[1], axis=1)[0]
    pred_class = inv_class_dict[pred_label]
    pred_concepts = np.where(pred[0] >= 0.5)
    if verbose:
        print(f'Predicted Concepts: {pred_concepts[1]}')
        print(f'Predicted Class: {pred_class}')
    attention = np.squeeze(get_attention(model, 'attn_score', test_input))
    pred_attn = attention[pred_concepts[1]]
    pred_text = concepts_text[pred_concepts[1]]

    #     plt.rcdefaults()
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(6, 5))

    if verbose:
        print(f'Predicted concept text before attention: {concepts_text[pred_concepts[1]]}')
        print(f"Scores before attention {np.squeeze(pred[0])[pred_concepts[1]]}")

    #
    # Negation to get maximum values
    ind = np.argsort(-attention)
    # min k such that there is more than 50% drop from the initial concept
    k = 0
    decrease_factor = 0.5
    highest_attn = attention[ind[k]]
    while highest_attn * decrease_factor < attention[ind[k]]:
        k += 1
    pred_text_after_attn = concepts_text[ind[:k]]
    if verbose:
        print(f"We have {k} values that are close to the maximum")
        print(f"top {k} values after attention: {attention[ind[:k]]}")
        print(f"top {k} after attention: {pred_text_after_attn}")

    pred_attn = attention[ind[:k]]
    y_pos = np.arange(len(pred_text_after_attn))
    ax.barh(y_pos, pred_attn, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pred_text_after_attn, fontsize=16)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Concept Score', fontsize=20)
    ax.set_title(f'Predicted Activity: {pred_class}', fontsize=20)
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()

    return pred, pred_label, pred_concepts, pred_attn


def to_conf_mtx(rows: List[str]) -> np.ndarray:
    mtx = [[int(elem) for elem in row.split()] for row in rows]
    mtx = np.array(mtx)

    return mtx


def calc_precision(conf_mtx: np.ndarray):
    tn = conf_mtx[0][0]
    fp = conf_mtx[0][1]
    fn = conf_mtx[1][0]
    tp = conf_mtx[1][1]

    return tp / (tp + fp)


def calc_f1(conf_mtx: np.ndarray):
    tn = conf_mtx[0][0]
    fp = conf_mtx[0][1]
    fn = conf_mtx[1][0]
    tp = conf_mtx[1][1]

    precision = calc_precision(conf_mtx)
    recall = tp / (tp + fn)

    return 2 * precision * recall / (precision + recall)


# Parse training log file
def parse_output_file(path: str, has_concepts: bool, binary=False):
    pattern = r"""
Results for model at time (\d+) *
Accuracy : (\d+\.\d+) *
F1-score : (\d+\.\d+) *"""
    if not binary:
        pattern = pattern + r"""
\[\[([\d ]+)\] *
 \[([\d ]+)\] *
 \[([\d ]+)\] *
 \[([\d ]+)\] *
 \[([\d ]+)\]\] *"""
    else:
        pattern = pattern + r"""
\[\[([\d ]+)\] *
 \[([\d ]+)\]\] *"""
    if has_concepts:
        pattern = pattern + r"""
\[\[([\d ]+)\] *
 \[([\d ]+)\]\] *
(\d+\.\d+) *"""

    str_file = open(path, 'r').read()
    results = regex.findall(pattern, str_file)

    conf_mtx_start = 3
    conf_mtx_end = 4 if binary else 7

    mid_i = 0
    accuracy_i = 1
    f1_i = 2
    conf_mtx_i = list(range(conf_mtx_start, conf_mtx_end + 1))
    conf_mtx_c_i = [conf_mtx_end + 1, conf_mtx_end + 2]
    concept_acc_i = conf_mtx_end + 3

    mids, accuracies, f1s, conf_mtxs, conf_mtx_cs, concept_accs, concept_f1, concept_precision = [], [], [], [], [], [], [], []
    for result in results:
        mids.append(result[mid_i])
        accuracies.append(float(result[accuracy_i]))
        f1s.append(float(result[f1_i]))
        conf_mtxs.append(to_conf_mtx([result[i] for i in conf_mtx_i]))
        if has_concepts:
            conf_mtx_c = to_conf_mtx([result[i] for i in conf_mtx_c_i])
            conf_mtx_cs.append(conf_mtx_c)
            concept_accs.append(float(result[concept_acc_i]))
            concept_f1.append(calc_f1(conf_mtx_c))
            concept_precision.append(calc_precision(conf_mtx_c))

    sol = {
        "model_id": mids,
        "accuracy": accuracies,
        "f1": f1s,
        "conf_mtx": conf_mtxs,
        "conf_mtx_c": conf_mtx_cs,
        "concept_acc": concept_accs,
        "concept_f1": concept_f1,
        "concept_precision": concept_precision,
    }
    return sol


# Parse transformer log file
def parse_transformer_output(path: str, binary=False):
    pattern = r"""
Model finished with accuracy: (\d+\.\d+), macro-f1: (\d+\.\d+) *
Confusion matrix: *"""
    if not binary:
        pattern = pattern + r"""
\[\[([\d ]+)\] *
 \[([\d ]+)\] *
 \[([\d ]+)\] *
 \[([\d ]+)\] *
 \[([\d ]+)\]\] *"""
    else:
        pattern = pattern + r"""
\[\[([\d ]+)\] *
 \[([\d ]+)\]\] *"""

    str_file = open(path, 'r').read()
    results = regex.findall(pattern, str_file)

    conf_mtx_start = 2
    conf_mtx_end = 3 if binary else 6

    accuracy_i = 0
    f1_i = 1
    conf_mtx_i = range(conf_mtx_start, conf_mtx_end + 1)

    accuracies, f1s, conf_mtxs = [], [], []
    for result in results:
        accuracies.append(float(result[accuracy_i]))
        f1s.append(float(result[f1_i]))
        conf_mtxs.append(to_conf_mtx([result[i] for i in conf_mtx_i]))

    sol = {
        "accuracy": accuracies,
        "f1": f1s,
        "conf_mtx": conf_mtxs,
    }
    return sol


# Reads CoDEx stored data as a dataframe of data-point results and a array of explanations
def read_pkl_data(pkl_result_path: str) -> (pd.DataFrame, np.ndarray):
    luke_output = pd.read_pickle(pkl_result_path)

    labels_keys = ['id', 'label', 'concepts']
    labels_dict = {key: luke_output[key] for key in labels_keys}
    labels_df_filtered = pd.DataFrame.from_dict(labels_dict)

    return labels_df_filtered, luke_output["explanations"]


# Load pre-computed extracted features
def fetch_extracted_features(labels: pd.DataFrame, features_dir: str) -> (pd.DataFrame, np.ndarray):
    try:
        labels = pd.read_pickle('labels_100.pkl')
        X = np.load('data.npy')

    except Exception:

        X = []
        for id in labels['id']:
            feature_path = os.path.join(features_dir, id + '.npy')
            if os.path.isfile(feature_path):
                vec = np.load(feature_path).T
                X.append(vec)
            else:
                labels = labels[labels['id'] != id]
                print(f"Id {id} not found")

        labels = labels.reset_index(drop=True)
        labels.to_pickle('labels_100.pkl')
        X = np.stack(X, axis=0)
        np.save('data.npy', X)

    return labels, X


# Returns a iterator which on each iteration returns the appropriate training/val data
def cross_val_iterator(*arrays: np.ndarray, cross_val_k: int) -> Iterator[Tuple[np.ndarray, ...]]:
    for val_fold in range(cross_val_k):
        iter_results = []
        for arr in arrays:
            train_split = []
            for train_fold in range(cross_val_k):
                if train_fold != val_fold:
                    train_split.append(arr[train_fold::cross_val_k])
            train_split = np.concatenate(train_split, axis=0)
            val_split = arr[val_fold::cross_val_k]
            iter_results.extend([train_split, val_split])
        yield tuple(iter_results)


# Unzip the dataset structure into tensors
def ds_unzip(ds: tf.data.Dataset, use_concepts: bool):
    X_vals, y_vals, c_vals = [], [], []

    for X, y in ds:
        X_vals.append(X)
        if type(y) is dict:
            c_vals.append(y['c_probs'])
            y_vals.append(y['probs'])
        else:
            y_vals.append(y)
    X_vals = tf.concat(X_vals, axis=0)
    y_vals = tf.concat(y_vals, axis=0)
    if use_concepts:
        c_vals = tf.concat(c_vals, axis=0)

    return X_vals, y_vals, c_vals


# Predict the concepts values after attention
def attn_prediction(model, input):
    pred = get_attention(model, 'attention_weights', input)
    return (pred + 1) / 2


# Return the appropriate dictionary structure from data
def get_concept_vector_dict(explanations, concept_preds, true_labels):
    output = {
        "probs": concept_preds,
        "explanations": explanations,
        "labels": true_labels,
    }
    return output


def load_labels_and_features(labels_df_filtered, features_dir):
    try:
        labels = pd.read_pickle('labels_100.pkl')
        X = np.load('data.npy')

    except:
        labels = labels_df_filtered.copy()
        print("Reading fresh version of the data")

        X = []
        for id in labels_df_filtered['id']:
            feature_path = os.path.join(features_dir, id + '.npy')
            if os.path.isfile(feature_path):
                vec = np.load(feature_path).T
                X.append(vec)

            else:
                labels = labels[labels['id'] != id]
                print(f"Id {id} not found")

        labels = labels.reset_index(drop=True)
        labels.to_pickle('labels_100.pkl')
        X = np.stack(X, axis=0)
        np.save('data.npy', X)

    print(X.shape)
    print(labels)
    return X, labels
