import torch
import pandas as pd
import os
from torch.utils.data import Dataset
from skimage import io
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Currently only rotation as that's the only distribution shift that can occur
random_transforms = transforms.Compose([
    transforms.RandomRotation((-90, 90))
])

# Rotated MNIST Data
class MNISTData(Dataset):
    def __init__(self, csv_file, root_dir, use_random_transforms=False):
        """
        Args:
            csv_file (string): Path to the csv file with label annotations.
            root_dir (string): Directory with all the images.
            use_random_transforms (bool): Should a random transformation be applied to an image.
        """
        self.data = list()
        self.root_dir = root_dir
        self.mnist_digits = pd.read_csv(csv_file)
        if use_random_transforms:
            self.custom_transform = random_transforms
        else:
            self.custom_transform = transforms.Compose([])

    def __len__(self):
        return len(self.mnist_digits)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, str(idx)+'.jpg')
        image = io.imread(img_name)
        label = int(self.mnist_digits.iloc[idx, 1])
        image = transform(image)
        image = self.custom_transform(image)

        return image.float(), label-1


def load_data(root_dir='.', data_type='standard', train_batch_size=64, test_batch_size=1, use_train_transforms=False):
    if data_type == 'standard':
        image_dir = root_dir + '/data/digits_1_to_4'
        train_ds = MNISTData(image_dir +'/train/labels.csv', image_dir +'/train', use_train_transforms)
        test_ds = MNISTData(image_dir+'/test/labels.csv', image_dir+'/test')
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=train_batch_size)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size)
    else:
        image_dir = root_dir + '/data/rotated_test_set_1_to_4'
        test_ds = MNISTData(image_dir + '/labels.csv', image_dir)
        train_loader = None
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size)
    return train_loader, test_loader


