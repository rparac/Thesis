import csv
from typing import Any, List, Dict

import pandas as pd

file_from = "/jupyter-notebooks/new_concept_grouping_lot_of_work.csv"
file_to = "/jupyter-notebooks/grouping_with_new_hyperparameters.csv"


def csv_to_dict(f) -> Dict[str, List[Any]]:
    ind_to_label = dict()
    dict1 = dict()
    with open(f) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                for i, label in enumerate(row):
                    ind_to_label[i] = label
                    dict1[label] = []

                line_count += 1
            else:
                for i, val in enumerate(row):
                    label = ind_to_label[i]
                    dict1[label].append(val)
                line_count += 1
        print(f'Processed {line_count} lines.')
    return dict1


dict1 = csv_to_dict(file_from)
dict2 = csv_to_dict(file_to)

indexes = []
for text in dict2['text']:
    try:
        indexes.append(dict1['text'].index(text))
    except ValueError:
        print(text)
values = [dict1['manual_id'][ind] for ind in indexes]
dict2['manual_id'] = values
print(dict1)

df2 = pd.DataFrame.from_dict(dict2)
new_df = df2[['manual_id', 'final_id', 'group_id', 'simple_prune_id', 'freq', 'text']]

new_df.to_csv("store_values.csv")
