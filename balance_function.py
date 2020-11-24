import numpy as np
import pickle
import torch

def count_dataset(spilt_data):
    count = []
    for i in range(len(spilt_data)):
        count.append(len(spilt_data[i]))
    return np.array(count)

def sep_data(dataset,final_out):
    train = []
    label = []
    for i in range(final_out):
        train.append([])
        label.append([])
    for i in range(len(dataset)):
        structure,ans = dataset.data_inputs[i],dataset.data_labels[i]
        train[int(ans)].append(structure)
        label[int(ans)].append(ans)
    return train,label

def data_append(spilt_data,spilt_label):
    # data append part
    train_data = []
    train_label = []
    count = count_dataset(spilt_data)
    max_try = 0
    while not np.all(count==0):
        print("Hello")
        i = np.random.choice(range(len(count)))
        max_try += 1
        if max_try > 10000:
            print("max_exceed")
            break
        j = count[i]-1
        if j < 0:  continue
        train_data.append(spilt_data[i][j])
        train_label.append(label[i][j])
        count[i] -= 1
    return train_data,train_label


def balance(dataset,final_out,outlier = []):
    spilt_data,label = sep_data(dataset,final_out)

    return data_append(spilt_data,label)


def balance_final(dataset,final_out,outlier = []):
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

        train_data,train_label = balance(dataset,28)
        print(len(train_data),len(train_label))
