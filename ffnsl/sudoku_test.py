import json
import math
import subprocess

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

from concept_processing.asp.asp_solver import clingo_solve
from concept_processing.classification.example_generator import ClassificationExampleGenerator
from ffnsl.sudoku_train import perform_feature_extraction, row_col_from_idx
from ffnsl.sudoku_4x4.feature_extractor.dataset import load_data as load_data4x4
from ffnsl.sudoku_9x9.feature_extractor.dataset import load_data as load_data9x9
from ffnsl.sudoku_9x9.feature_extractor.network import MNISTNet
from ffnsl.sudoku_4x4.feature_extractor.network import NewMNISTNet

from ffnsl.sudoku_4x4.sudoku_dataset import load_sudoku_data as load_sudoku_data4x4
from ffnsl.sudoku_9x9.sudoku_dataset import load_sudoku_data as load_sudoku_data9x9

project_dir = '/home/rp218/luke-for-roko'
sudoku_num_of_digits = 4
sudoku_board_size = f'{sudoku_num_of_digits}x{sudoku_num_of_digits}'  # or 9x9
sudoku_dir = f'sudoku_{sudoku_board_size}'

saved_model_path = {
    '4x4': f'{sudoku_dir}/feature_extractor/saved_model/model.pth',
    '9x9': f'{sudoku_dir}/feature_extractor/saved_model/model.pth',
}

def data_label_iterator(structured_test_file):
    for train_idx, row in structured_test_file.iterrows():
        yield row["puzzle"].split(), 1 - row["label_code"]
# if tensor or not tensor

tmp_file_loc = "/tmp/ffnsl-example_context.lp"
output_dir = f'{project_dir}/ilasp/classification/{sudoku_dir}'
use_generator_based_nn = False
net_name1 = "edl_gen" if use_generator_based_nn else "softmax"
net_name2 = "edl_gen" if use_generator_based_nn else "standard"
learned_file_tmp = f'{output_dir}/run_noise_experiments/run_noise_{net_name2}_repeat{{}}_{{}}.txt'

# Do we evaluate the test data with 100% accurate information
eval_structured = False
cache_dir= f"{sudoku_dir}/results/prob_nsl/{'structured' if eval_structured else 'unstructured'}/{net_name1}"


# TODO: replace prediction_dict with true predictions
if __name__ == "__main__":
    structured_test_file = pd.read_csv(f'{sudoku_dir}/data/structured_data/small/test.csv')
    structured_test_file = structured_test_file.astype({"label": "category"})
    structured_test_file["label_code"] = structured_test_file['label'].cat.codes

    non_perturbed_preds = perform_feature_extraction("standard", sudoku_dir, sudoku_board_size, use_edl_gen=use_generator_based_nn)
    perturbed_preds = perform_feature_extraction("rotated", sudoku_dir, sudoku_board_size, use_edl_gen=use_generator_based_nn)

    if sudoku_board_size == '4x4':
        _, sud_test_load = load_sudoku_data4x4(base_dir=f'{sudoku_dir}')
    else:
        _, sud_test_load = load_sudoku_data9x9(base_dir=f'{sudoku_dir}')

    repeats = list(range(1, 11))

    result_dict = {}
    # for noise_pct in range(70, 101, 100):
    for noise_pct in range(0, 101, 10):
        acc_list = []
        for repeat in repeats:
            print(f"Experimenting with noise_pct={noise_pct}, repeat={repeat}")
            y_true, y_pred = [], []
            test_loader = data_label_iterator(structured_test_file) if eval_structured else sud_test_load
            for batch_idx, (data, target) in enumerate(test_loader):
                if type(target) is torch.Tensor:
                    target = target.item()

                preds = []
                ctxs_str = []
                ex_generator = ClassificationExampleGenerator(categories=["valid", "invalid"])
                for idx, cell in enumerate(data):
                    if type(cell) is str:
                        cell = int(cell)
                    else:
                        num_perturbed_examples = math.floor((noise_pct / 100) * len(test_loader))
                        prediction_dict = perturbed_preds if batch_idx < num_perturbed_examples else non_perturbed_preds

                        cell = np.argmax(prediction_dict[f"{cell.item()}.jpg"]) + 1 if cell.item() != 0 else 0
                    if cell != 0:
                        pred = np.zeros(sudoku_num_of_digits)
                        pred[cell - 1] = 1
                        preds.append(pred)
                        row_num, col_num = row_col_from_idx(idx, sudoku_num_of_digits)
                        ctxs_str.append(f"value({row_num}, {col_num}, {{}}).")
                ex_generator.parse(preds, ctxs_str, target)
                example_info = ex_generator.get_context(out_offset=1)

                with open(tmp_file_loc, 'w') as f:
                    f.write(example_info)
                    f.write('\n')

                learned_file = learned_file_tmp.format(noise_pct, repeat)
                sol = clingo_solve(f"{output_dir}/background.lp", tmp_file_loc, learned_file)
                outcome = int("selected(invalid)" in sol[0])
                y_true.append(target)
                y_pred.append(outcome)

            acc_list.append(accuracy_score(y_true, y_pred))

        accs = np.array(acc_list)
        std = np.std(accs)
        se = std / math.sqrt(accs.shape[0])
        print(f"Accuracy: noise={noise_pct}%, mean: {np.mean(accs)}, standard error {se}")
        acc_results = {"mean": np.mean(accs), "std": std, "std_err": se, "raw": acc_list}
        result_dict[f"noise_pct_{noise_pct}"] = {"accuracy": acc_results}

    standard_dict = {'noise_pct_0': result_dict['noise_pct_0']}
    with open(cache_dir + '/standard.json', 'w') as cache_out:
        cache_out.write(json.dumps(standard_dict))

    del result_dict['noise_pct_0']
    with open(cache_dir + '/rotated.json', 'w') as cache_out:
        cache_out.write(json.dumps(result_dict))
