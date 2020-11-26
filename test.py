import numpy as np
import pickle
import torch.nn.functional
import matplotlib.pyplot as plt
import balance_function as bf
import data_loader
import function_training
import function_list

import crystalsystem
import os

# prepare neural network
csnum = 7
validate_size = 0.1
num_bands = 100
tmp = []
for hs in range(48):
    tmp.append(hs)
hs_indices = tmp
#hs_indices = [0, 1, 3, 4, 5, 7, 8, 13, 31, 34, 37]  # 11 hs points in Brillouin zone out of 40

cs_sizes = crystalsystem.crystalsystem_sizes()
output_size = cs_sizes[csnum - 1] - cs_sizes[csnum - 2] + 1 if csnum > 1 else 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.nn.Sequential(
    #torch.nn.LeakyReLU(),
    torch.nn.Linear(len(hs_indices)*num_bands, 128),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(128, 64),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(64, output_size),
    torch.nn.LeakyReLU(),
)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)
criterion = torch.nn.CrossEntropyLoss()
with open ("data.pickle","rb") as f:
    dataset = pickle.load(f)

# data cut off and shuffle
#data_in,data_out = bf.data_cutoff(dataset,output_size,cut_off=70)
#data_loader.update_dataset(dataset,data_in,data_out)
# spilt dataset
train_dataset,test_dataset = data_loader.spilt_train_test_dataset(dataset)
#train_dataset,test_dataset = data_loader.advanced_spilt_train_test_dataset(dataset,output_size)
# balance train part
#train_in,train_out = bf.balance_avg(train_dataset,output_size)
#print(f"Before balance:\n{bf.view_count(train_dataset,output_size)}")
#data_loader.update_dataset(train_dataset,train_in,train_out)
#print(f"After balance:\n{bf.view_count(train_dataset,output_size)}")
validate_loader = data_loader.get_validate_loader(test_dataset,32)
train_loader = data_loader.get_train_loader(train_dataset,32)

def validate_one_epoch(device, model, criterion, validate_loader):
    model.eval()
    num_validate = len(validate_loader.sampler.indices)
    if num_validate == 0:
        print("number of data is 0")
        return -1, -1
    val_loss = 0.
    num_correct = 0
    for b, (batch_input, batch_label) in enumerate(validate_loader):
        for i in range(len(batch_input)):
            # read data
            data_input, data_label = batch_input[i], batch_label[i]
            print(data_input)
            exit()
            data_input, data_label = data_input.to(device), data_label.to(device)
            # feed
            output = model(data_input).view(1, -1)
            # record fitness
            val_loss += criterion(output, data_label).item()
            if torch.max(output, 1)[1] == data_label:
                num_correct += 1
        print("\r\tvalidate batch:{}/{}".format(b, len(validate_loader)), end="")
    val_loss /= num_validate
    num_correct /= num_validate
    return round(val_loss, 4), round(num_correct*100, 4)

loss,acc = validate_one_epoch(device, model, criterion, validate_loader)
