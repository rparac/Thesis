import os
import random
import shutil

new_location = '/home/rp218/projects/thesis/bird_flowers_ds'

description_file = os.path.join(new_location, 'explanations.csv')

with open(description_file, 'w') as f:
    new_dir_birds = os.path.join(new_location, "birds/text")
    for file in os.listdir(new_dir_birds):
        id = file.replace(".txt", "")
        file_txt = open(f"{new_dir_birds}/{file}").readline().replace('\n', "")

        f.write(f'{id},bird,{file_txt}\n')

    new_dir_text = os.path.join(new_location, "flowers/text")
    for file in os.listdir(new_dir_text):
        id = file.replace(".txt", "")
        file_txt = open(f"{new_dir_text}/{file}").readline().replace('\n', "")
        f.write(f'{id},flower,{file_txt}\n')
