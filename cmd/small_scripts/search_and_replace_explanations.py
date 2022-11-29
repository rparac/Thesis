"""
Simple script to display to replace numbers with explanations in a solution provided by FastLAS.
This aids interpretability.
"""
import pickle
import re

import numpy as np

# pam_storage = '/home/rp218/luke-for-roko/cmd/birds_flowers_final_dict_new_codex.pkl'
pam_storage = '/home/rp218/luke-for-roko/Extracted_Concepts/final_dict_new_codex.pkl'
sol_location = '/home/rp218/luke-for-roko/ilasp/classification/baseball_model_attn/new_run_0.out'
output_location = '/home/rp218/luke-for-roko/ilasp/classification/baseball_model_attn/solution.txt'

if __name__ == '__main__':
    c_dict = pickle.load(open(pam_storage, 'rb'))

    ids = c_dict['id']
    labels = c_dict['label']
    concepts = c_dict['concepts']
    explanations = c_dict['explanations']
    categories = np.unique(labels)

    file_str = open(sol_location, 'r').read()

    for i, explanation in enumerate(explanations):
        file_str = file_str.replace(f'concept({i})', f'concept("{explanation}")')

    with open(output_location, 'w') as f:
        f.write(file_str)
