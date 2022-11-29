"""
Runs the bird-data classifier training
This script is designed to be run on the DoC machines.
The dataset used with this code is stored at: https://gitlab.doc.ic.ac.uk/rp218/birds_blowers_ds
"""

import os
import re
import time
from typing import List

import keras.preprocessing.image_dataset
import pandas as pd
import tensorflow as tf
from keras.applications import resnet_v2

from nn.model_architectures import model_image_classification, model_image_classification_from_concepts
from nn.utils import ds_unzip, calculate_metrics_binary, \
    train_model, train_concept_model, calculate_concept_metrics_binary


def preprocess_data(images, labels):
    return resnet_v2.preprocess_input(images), labels


def to_image_ids(file_paths: List[str]) -> List[str]:
    return [re.match(r".*/([a-zA-Z0-9_]+)\.jpg", file_path).group(1) for file_path in file_paths]


def idx_lookup(df: pd.DataFrame, row_key: str, value: str) -> int:
    return df[row_key][df[row_key] == value].index[0]


network_name = "image_classifier_network"
large_data_dir = "/vol/bitbucket/rp218/Thesis_Data"
model_dir = f"{large_data_dir}/Models"

use_concepts = True
concepts_only_training = True
n_concepts = 78
dir = '/vol/bitbucket/rp218/bird_flowers_ds'
dataset_train_split = 1500
dataset_val_point = 1700

dataset = keras.preprocessing.image_dataset.image_dataset_from_directory(dir, image_size=(224, 224), batch_size=1,
                                                                         shuffle=False)

X_data, y_data, _ = ds_unzip(dataset, use_concepts=False)
X_data = resnet_v2.preprocess_input(X_data)

# Read the pickle file
project_dir = '/vol/bitbucket/rp218/luke-for-roko'
path = os.path.join(project_dir, "Extracted_Concepts/birds_flowers_final_dict_new_codex.pkl")
luke_output = pd.read_pickle(path)

labels_keys = ['id', 'label', 'concepts']
labels_dict = {key: luke_output[key] for key in labels_keys}
labels = pd.DataFrame.from_dict(labels_dict)

ds_img_ids = to_image_ids(dataset.file_paths)
concept_list_ids = [idx_lookup(labels, "id", img_id) for img_id in ds_img_ids]
concept_list = [tf.convert_to_tensor(labels["concepts"].iloc[id]) for id in concept_list_ids]
y_data = [{"c_probs": c, "probs": y} for c, y in zip(concept_list, y_data)]


def y_generator_fun():
    for y in y_data:
        yield y


X_dataset = tf.data.Dataset.from_tensor_slices(X_data)
y_dataset = tf.data.Dataset.from_generator(y_generator_fun, output_signature=({
    "c_probs": tf.TensorSpec(shape=(n_concepts,)),
    "probs": tf.TensorSpec(shape=())
}))
preprocesed_dataset = tf.data.Dataset.zip((X_dataset, y_dataset)).batch(1)

X, y, c = ds_unzip(preprocesed_dataset, use_concepts=True)

# Manually shuffling because tf dataset shuffle doesn't work
indices = tf.range(start=0, limit=tf.shape(X)[0], dtype=tf.int32)
shuffled_indices = tf.random.shuffle(indices)

X = tf.gather(X, shuffled_indices)
y = tf.gather(y, shuffled_indices)
c = tf.gather(c, shuffled_indices)

# X_train, y_train, c_train = ds_unzip(preprocesed_dataset, use_concepts=True)
# X_test, y_test, c_test = ds_unzip(shuffled_dataset, use_concepts=True)
X_train = X[:dataset_train_split]
X_val = X[dataset_train_split:dataset_val_point]
X_test = X[dataset_val_point:]
y_train = y[:dataset_train_split]
y_val = y[dataset_train_split:dataset_val_point]
y_test = y[dataset_val_point:]
c_train = c[:dataset_train_split]
c_val = c[dataset_train_split:dataset_val_point]
c_test = c[dataset_val_point:]

t = int(time.time())

if use_concepts and not concepts_only_training:
    # model, history = train_concept_model_sequential(model_tuple, X_train, y_train, c_train, X_test, y_test, c_test,
    #                                                 model_dir, t, n_concepts,
    #                                                 network_name, epochs=100, batch_size=32)
    _, model = model_image_classification(num_classes=1, n_concept_layer=n_concepts)
    model, history = train_concept_model(model, X_train, y_train, c_train, X_val, y_val, c_val, model_dir, t,
                                         n_concepts, network_name, epochs=100, batch_size=32)

    cf_matrix, accuracy, macro_f1, mismatch, y_pred, cf_concepts, accuracy_concepts \
        = calculate_concept_metrics_binary(model, X_test, y_test, c_test.numpy())

    print(f"Results for model at time {t}")
    print('Accuracy : {}'.format(accuracy))
    print('F1-score : {}'.format(macro_f1))
    print(cf_matrix)
    print(cf_concepts)
    print(accuracy_concepts)

elif use_concepts and concepts_only_training:
    model = model_image_classification_from_concepts(num_classes=1, n_concepts=n_concepts)
    model, H = train_model(model, c_train, y_train, c_val, y_val, model_dir, t,
                           batch_size=32, epochs=100, name="concept_only_birds_flowers")

    cf_matrix, accuracy, macro_f1, mismatch, y_pred, = calculate_metrics_binary(model, c_test, y_test)
    print(f"Results for model at time {t}")
    print('Accuracy : {}'.format(accuracy))
    print('F1-score : {}'.format(macro_f1))
    print(cf_matrix)

else:
    _, model = model_image_classification(num_classes=1, n_concept_layer=n_concepts, use_concepts=False)
    model, H = train_model(model, X_train, y_train, X_val, y_val, model_dir, t,
                           batch_size=32, epochs=100, name="concept_free_birds_flowers")

    cf_matrix, accuracy, macro_f1, mismatch, y_pred, = calculate_metrics_binary(model, X_test, y_test)
    print(f"Results for model at time {t}")
    print('Accuracy : {}'.format(accuracy))
    print('F1-score : {}'.format(macro_f1))
    print(cf_matrix)

# # TODO: remove duplication
# if use_concepts:
#     c_pred, y_pred = full_model.predict(X_test)
#     y_pred = y_pred.squeeze()
#     y_pred = (y_pred >= 0.5).astype(int)
#     y_true = y_test.numpy()
#
#     cf_matrix = confusion_matrix(y_true, y_pred)
#     accuracy = accuracy_score(y_true, y_pred)
#
#     print(f"Possible classes are {list(enumerate(dataset.class_names))}")
#     print("Confusion matrix:")
#     print(cf_matrix)
#     print(f"Accuracy is: {accuracy}")
#
#     c_test = c_test.numpy().flatten()
#     c_pred = c_pred.flatten()
#     c_pred[c_pred <= 0.5] = 0
#     c_pred[c_pred > 0.5] = 1
#     cf_concepts = confusion_matrix(c_test, c_pred)
#     precision = precision_score(c_test, c_pred)
#     print("Confusion matrix:")
#     print(cf_concepts)
#     print(f"Precision is: {precision}")
#
#     # Store to vector
#     concept_preds, _ = full_model.predict(train_ds)
#     true_train_labels = labels["label"].iloc[concept_list_ids[:1700]].values
#     print(concept_preds.shape)
#     print(true_train_labels.shape)
#
#     explanations = luke_output['explanations']
#     output = {
#         "probs": concept_preds,
#         "explanations": explanations,
#         "labels": true_train_labels,
#     }
#     pickle.dump(output, open("for_fastlas_examples_train.pkl", 'wb'))
#
#     concept_preds, _ = full_model.predict(test_ds)
#     true_train_labels = labels["label"].iloc[concept_list_ids[1700:]].values
#     print(concept_preds.shape)
#     print(true_train_labels.shape)
#
#     explanations = luke_output['explanations']
#     output = {
#         "probs": concept_preds,
#         "explanations": explanations,
#         "labels": true_train_labels,
#     }
#
#     pickle.dump(output, open("for_fastlas_examples_test.pkl", 'wb'))
#
#
# else:
#     y_pred = full_model.predict(X_test).squeeze()
#     y_pred = (y_pred >= 0.5).astype(int)
#     y_true = y_test.numpy()
#
#     cf_matrix = confusion_matrix(y_true, y_pred)
#     accuracy = accuracy_score(y_true, y_pred)
#
#     print(f"Possible classes are {list(enumerate(dataset.class_names))}")
#     print("Confusion matrix:")
#     print(cf_matrix)
#     print(f"Accuracy is: {accuracy}")
