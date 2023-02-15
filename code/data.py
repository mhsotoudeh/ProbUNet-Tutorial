import pickle

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class GRFDataset(Dataset):
    def __init__(self, inputs, truths, img_transform=None, truth_transform=None, metas=None):
        self.inputs = inputs
        self.truths = truths
        self.img_transform = img_transform
        self.truth_transform = truth_transform
        self.metas = metas

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        image = self.inputs[idx]
        truth = self.truths[idx]
        
        if self.img_transform:
            image = self.img_transform(image)
        if self.truth_transform:
            truth = self.truth_transform(truth)

        if self.metas is None:
            return image, truth
        else:
            meta = self.metas[idx]
            return image, truth, meta



def prepare_data(datafile, size=None, shuffle=False, normalization='standard', eps=1e-6, metafile=None):
    with open(datafile, 'rb') as f:
        truths = np.load(f)
        inputs = np.load(f)

    if size is None:
        size = len(inputs)

    if shuffle is True:
        p = np.random.permutation(len(inputs))
        inputs, truths = inputs[p], truths[p]

    # Select required data
    inputs  = inputs[:size]
    truths = truths[:size]

    # Load Meta
    metas = None
    if metafile is not None:
        with open(metafile, 'rb') as f:
            metas = pickle.load(f)
        if shuffle is True:
            metas = np.array(metas)[p].tolist()
            

    # Define Normalization Transforms
    if normalization == 'standard':
        # Calculate mean and std for standardization
        input_mean, input_std = np.mean(inputs), np.std(inputs)
        truth_mean, truth_std = np.mean(truths), np.std(truths)


        # Define Transforms
        input_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=input_mean, std=input_std)
        ])

        truth_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=truth_mean, std=truth_std)
        ])


        # Define Inverse Transforms
        inv_input_trans = transforms.Compose([
            transforms.Normalize(mean=0.0, std=1.0/input_std),
            transforms.Normalize(mean=-input_mean, std=1.0)
        ])

        inv_truth_trans = transforms.Compose([
            transforms.Normalize(mean=0.0, std=1.0/truth_std),
            transforms.Normalize(mean=-truth_mean, std=1.0)
        ])


    elif normalization is None:
        # Define Transforms
        input_trans = transforms.ToTensor()
        truth_trans = transforms.ToTensor()

        # Define Inverse Transforms
        inv_input_trans = transforms.Normalize(mean=0.0, std=1.0)
        inv_truth_trans = transforms.Normalize(mean=0.0, std=1.0)


    transdict = {
        'input_transform': input_trans,
        'truth_transform': truth_trans,
        'inv_input_transform': inv_input_trans,
        'inv_truth_transform': inv_truth_trans
    }


    # Create Datasets
    train_data = GRFDataset(inputs, truths, img_transform=input_trans, truth_transform=truth_trans, metas=metas)

    return train_data, transdict
