import math
import pickle

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

from concept_processing.classification.example_generator import get_concept_bottleneck_context, predict

eval_file_tmp = '../../ilasp/classification/baseball_model_attn/ctx_{}.las'
background_loc = '../../ilasp/classification/baseball_model_attn/background.lp'
conc_vec_storage_tmp = '/home/rp218/luke-for-roko/cmd/fastlas_examples/for_fastlas_examples_test_attn_{}.pkl'
sol_file_tmp = '/home/rp218/luke-for-roko/ilasp/classification/baseball_model_attn/new_run_{}.out'

if __name__ == '__main__':

    to_digit = {
        "strike": 0,
        "foul": 1,
        "out": 2,
        "play": 3,
        "ball": 4,
    }

    accs = []

    for i in range(0, 10):
        conc_vec_storage = conc_vec_storage_tmp.format(i)
        c_dict = pickle.load(open(conc_vec_storage, 'rb'))

        probs = c_dict["probs"]
        explanations = c_dict["explanations"]
        labels = c_dict["labels"]
        categories = np.unique(labels)

        n_explanations = len(explanations)
        y_pred, y_true = [], []

        for j, (nn_outcomes, label) in enumerate(zip(probs, labels)):
            ctx = get_concept_bottleneck_context(categories, nn_outcomes, label)
            sol_str = predict(ctx, sol_file_tmp.format(i), background_loc)

            y_pred.append(to_digit[sol_str])
            y_true.append(to_digit[label])

        c_matrix = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        accs.append(acc)
        print(c_matrix)
        print(acc)

    accs = np.array(accs)
    std = np.std(accs)
    se = std / math.sqrt(accs.shape[0])
    print(f"Accuracy: mean {np.mean(accs)}, standard deviation {std}, standard error {se} ")
