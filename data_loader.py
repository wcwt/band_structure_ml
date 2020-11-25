import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler



class AnyDataset(Dataset):
    def __init__(self, in_list_paths, json2inputlabel, validate_size, empty_class = False):
        if empty_class:
            self.validate_size = validate_size
            self.data_inputs = []
            self.data_labels = []
            self.len = 0
            self.split = 0
        else:
            self.validate_size = validate_size
            validate_inputs = []
            validate_labels = []
            train_inputs = []
            train_labels = []
            if not isinstance(in_list_paths, list):  # if only one path is parsed
                in_list_paths = [in_list_paths]
            for i, in_list_path in enumerate(in_list_paths):
                if os.stat(in_list_path).st_size == 0:  # if file is empty
                    continue
                file_paths = np.loadtxt(in_list_path, "U90", ndmin=1)
                split = int(validate_size*len(file_paths))
                for j, file_name in enumerate(file_paths):
                    with open(file_name, "r") as file:
                        data_json = json.load(file)
                    # json2inputlabel is a function parsed to init, which handles a json dict and give input and label
                    data_input_np, data_label_np = json2inputlabel(data_json)
                    if j <= split:
                        validate_inputs.append(torch.from_numpy(data_input_np).float())
                        validate_labels.append(torch.from_numpy(data_label_np).long())
                    else:
                        train_inputs.append(torch.from_numpy(data_input_np).float())
                        train_labels.append(torch.from_numpy(data_label_np).long())
                print("\r\tload: {}/{}".format(i, len(in_list_paths)), end="")
            print("\rload: {}".format(len(in_list_paths)))
            if len(train_inputs) == 0:
                raise OSError("input list file is empty")
            print(f"train: {len(train_inputs)}")
            print(f"validate: {len(validate_inputs)}")
            self.data_inputs = [*validate_inputs, *train_inputs]
            self.data_labels = [*validate_labels, *train_labels]

            self.len = len(self.data_inputs)
            self.split = int(validate_size * self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data_inputs[index], self.data_labels[index]

    def update_inform(self):
        self.len = len(self.data_inputs)
        self.split = int(self.validate_size * self.len)


def get_validate_train_loader(dataset, batch_size):
    validate_sampler = SubsetRandomSampler(range(dataset.split))
    train_sampler = SubsetRandomSampler(range(dataset.split, len(dataset)))
    validate_loader = DataLoader(dataset, batch_size=batch_size, sampler=validate_sampler)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    return validate_loader, train_loader

def get_train_loader(dataset, batch_size):
    train_sampler = SubsetRandomSampler(range(dataset.split, len(dataset)))
    print(f"train range {dataset.split} to {len(dataset)-1}")
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    return train_loader

def get_validate_loader(dataset, batch_size):
    validate_sampler = SubsetRandomSampler(range(dataset.split))
    print(f"test range {0} to {dataset.split}")
    validate_loader = DataLoader(dataset, batch_size=batch_size, sampler=validate_sampler)
    return validate_loader

def spilt_test_train_dataset(dataset,batch_size):
    test_dataset = AnyDataset("","",1,empty_class=True)
    test_dataset.data_inputs = dataset.data_inputs[:dataset.split]
    test_dataset.data_labels = dataset.data_labels[:dataset.split]
    test_dataset.update_inform()
    print("data size for test ",len(test_dataset.data_inputs))
    train_dataset = AnyDataset("","",0,empty_class=True)
    train_dataset.data_inputs = dataset.data_inputs[dataset.split:]
    train_dataset.data_labels = dataset.data_labels[:dataset.split:]
    train_dataset.update_inform()
    print("data size for train",len(train_dataset.data_inputs))
    return get_validate_loader(test_dataset, batch_size),get_train_loader(train_dataset,batch_size)
