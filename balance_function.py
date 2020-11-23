import numpy as np
import pickle
import torch

with open ("data.pickle","rb") as f:
    dataset = pickle.load(f)


def count_dataset(spilt_data):
    count = []
    for i in range(len(spilt_data)):
        count.append(len(spilt_data[i]))
    return np.array(count)

def sep_data(dataset,final_out):
    train = []
    count = []
    for i in range(final_out):
        train.append([])
        count.append(0)
    for i in range(len(dataset)):
        structure,ans = dataset[i][0],dataset[i][1]
        train[ans].append(structure)
        count[ans] += 1
    return train,np.array(count)

def balance(dataset,final_out,outlier = []):
    spilt_data,count = sep_data(dataset,final_out)
    # get average
    avg = 0
    num = 0
    # get average
    for i in range(len(count)):
        if (i in outlier) or (count[i] == 0):    continue
        avg += count[i]
        num += 1
    avg /= num
    avg = int(avg + 0.5)
    for i in range(len(count)):
        if count[i]==0: continue
        while count[i] < avg:
            add_index = np.random.choice(range(count[i]))
            spilt_data[i].append(spilt_data[i][add_index])
            count[i] += 1
        while count[i] > avg:
            del_index = np.random.choice(range(count[i]))
            del spilt_data[i][del_index]
            count[i] -= 1
    train_data = []
    train_label = []
    for i in range(len(count)):
        if count[i] == 0:   continue
        for ele in spilt_data[i]:
            train_data.append(ele)
            train_label.append(torch.tensor([i]))
    return train_data,train_label

if __name__ == '__main__':
    with open ("data.pickle","rb") as f:
        dataset = pickle.load(f)

    spilt_data,count = balance(dataset,27,outlier=[26])
    train_data = []
    train_label = []
    for i in range(len(count)):
        if count[i] == 0:   continue
        for ele in spilt_data[i]:
            train_data.append(ele)
            train_label.append(torch.tensor([i]))



