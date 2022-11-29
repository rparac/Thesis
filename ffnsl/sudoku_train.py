import argparse
import json
import math
import subprocess
import sys
from os.path import dirname, realpath

import numpy as np
import torch
import torch.nn as nn

from concept_processing.classification.example_generator import ClassificationExampleGenerator
from ffnsl.sudoku_4x4.feature_extractor.dataset import load_data as load_data4x4
from ffnsl.sudoku_9x9.feature_extractor.dataset import load_data as load_data9x9
from ffnsl.sudoku_9x9.feature_extractor.network import MNISTNet
from ffnsl.sudoku_4x4.feature_extractor.network import NewMNISTNet

from ffnsl.sudoku_4x4.sudoku_dataset import load_sudoku_data as load_sudoku_data4x4
from ffnsl.sudoku_9x9.sudoku_dataset import load_sudoku_data as load_sudoku_data9x9

# Add root directory to path
file_path = realpath(__file__)
file_dir = dirname(file_path)
parent_dir = dirname(file_dir)
root_dir = dirname(parent_dir)
sys.path.append(root_dir)

project_dir = '/home/rp218/luke-for-roko'
sudoku_num_of_digits = 9  # 9

saved_model_path = {
    '4x4': f'sudoku_4x4/feature_extractor/saved_model/model.pth',
    '9x9': f'sudoku_9x9/feature_extractor/saved_model/model.pth',
}


def get_nn(sudoku_board_size):
    if sudoku_board_size == '4x4':
        net = NewMNISTNet()
    else:
        net = MNISTNet()
    network_state_dict = torch.load(saved_model_path[sudoku_board_size])
    net.load_state_dict(network_state_dict)
    return net


def perform_feature_extraction(ds, sudoku_dir, sudoku_board_size, use_edl_gen=False):
    if use_edl_gen:
        cache_dir = f'{project_dir}/ffnsl/{sudoku_dir}/cache/digit_predictions/edl_gen'
        full_path = f'{cache_dir}/{ds}_test_set.json'
        predictions = json.loads(open(full_path).read())
        predictions = {k: np.array(v) for k, v in predictions.items()}
    else:
        # Load data
        if sudoku_board_size == "4x4":
            train_l, test_l = load_data4x4(root_dir=f'{sudoku_dir}/feature_extractor', data_type=ds)
        else:
            train_l, test_l = load_data9x9(root_dir=f'{sudoku_dir}/feature_extractor', data_type=ds)

        # Instantiate network and load trained weights
        net = get_nn(sudoku_board_size)
        # dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        dev = torch.device("cpu")
        net.to(dev)
        net.eval()

        # Initialise prediction dictionary
        predictions = {}

        # Perform forward pass on network
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_l):
                data.to(dev)
                output = net(data)
                softmax_fn = nn.Softmax(dim=1)
                softmax_output = softmax_fn(output)
                predictions[str(batch_idx) + '.jpg'] = torch.squeeze(softmax_output).numpy()

    return predictions


def row_col_from_idx(idx, sudoku_num_digits):
    row_number = math.ceil((idx + 1) / sudoku_num_digits)
    col_number = (idx + 1) % sudoku_num_digits
    if col_number == 0:
        col_number = sudoku_num_digits
    return row_number, col_number


use_generator_based_nn = False
# use edl_gen NN for MNIST digit classification; uses a standard one if false

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate guesses and evaluate solutions')
    parser.add_argument('--num-digits', type=int, nargs='?', default=sudoku_num_of_digits)
    parser.add_argument('--noise-pct', type=int, nargs='+', default=list(range(0, 101, 10)))
    parser.add_argument('--repeat-num', type=int, nargs='+', default=list(range(1, 11)))
    parser.add_argument('--generator-nn', action='store_true', default=use_generator_based_nn)
    args = parser.parse_args()

    use_generator_based_nn = args.generator_nn
    sudoku_num_of_digits = args.num_digits

    # config
    sudoku_board_size = f'{sudoku_num_of_digits}x{sudoku_num_of_digits}'
    sudoku_dir = f'sudoku_{sudoku_board_size}'
    output_dir = f'{project_dir}/ilasp/classification/{sudoku_dir}'
    net_name = "edl_gen" if use_generator_based_nn else "standard"
    output_run_file_tmp = f'{output_dir}/run_noise_{net_name}_repeat{{}}_{{}}.txt'

    noise_pcts = args.noise_pct
    repeats = list(args.repeat_num)

    non_perturbed_preds = perform_feature_extraction("standard", sudoku_dir, sudoku_board_size, use_edl_gen=use_generator_based_nn)
    perturbed_preds = perform_feature_extraction("rotated", sudoku_dir, sudoku_board_size, use_edl_gen=use_generator_based_nn)

    # repeats = [1]

    if sudoku_board_size == '4x4':
        sud_train_loaders, _ = load_sudoku_data4x4(base_dir=f'{sudoku_dir}', repeats=repeats)
    else:
        sud_train_loaders, _ = load_sudoku_data9x9(base_dir=f'{sudoku_dir}', repeats=repeats)

    # for noise_pct in range(0, 101, 10):
    for noise_pct in noise_pcts:
        print(f"Experimenting with noise_pct={noise_pct}")
        for repeat, train_loader in zip(repeats, sud_train_loaders):
            output_ex_file = f'{output_dir}/examples_{net_name}_{repeat}_{noise_pct}.las'
            print(f"Iteration {repeat}")
            ex_generator = ClassificationExampleGenerator(categories=["valid", "invalid"])
            num_perturbed_examples = math.floor((noise_pct / 100) * len(train_loader))
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx < num_perturbed_examples:
                    prediction_dict = perturbed_preds
                else:
                    prediction_dict = non_perturbed_preds

                preds = []
                ctxs_str = []
                for idx, cell in enumerate(data):
                    # In the original code, 0 is used as a default value. It would be better if it was -1 since
                    # 0.jpg does exist. However, it is not changed to keep the experiments as consistent as possible with
                    # Dan's
                    if cell.item() != 0:
                        preds.append(prediction_dict[f"{cell.item()}.jpg"])
                        row_num, col_num = row_col_from_idx(idx, sudoku_num_of_digits)
                        ctxs_str.append(f"value({row_num}, {col_num}, {{}}).")
                ex_generator.parse(preds, ctxs_str, target.item())

            with open(output_ex_file, 'w') as f:
                for example in ex_generator.get_examples(n_of_predicates=8, multilabel=False, out_offset=1):
                    f.write(example)
                    f.write('\n')

            output_run_file = output_run_file_tmp.format(noise_pct, repeat)
            with open(output_run_file, "w") as f:
                subprocess.run([
                    "FastLAS", "--nopl",
                    f"{output_dir}/background.lp",
                    f"{output_dir}/language_bias.las",
                    output_ex_file,
                    "--debug",
                ], stdout=f)

    print("Done")
