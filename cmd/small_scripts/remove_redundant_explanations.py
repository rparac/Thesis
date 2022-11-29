"""
Constructs the explanations.csv from CUB_birds_dataset
"""

import os

base_dir = '/home/rp218/projects/thesis/CUB_200_2011/text'

first_row = 'id,label,explanation'

if __name__ == "__main__":
    file_lines = [first_row]
    for bird_folder in os.listdir(base_dir):
        for bird_file in os.listdir(f"{base_dir}/{bird_folder}"):
            with open(f"{base_dir}/{bird_folder}/{bird_file}") as f:
                explanation = f.readline().strip('\n')
                image_id = bird_file.rstrip(".txt")
                file_lines.append(f"{image_id},{bird_folder},{explanation}")

    with open(f"{base_dir}/explanations.csv", 'w') as f:
        f.write('\n'.join(file_lines))


