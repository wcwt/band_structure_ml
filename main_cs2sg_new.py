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

dir = "report_graph/"
folder = dir + "cut_off_70_advicedVailder_balanced"
if not os.path.exists(folder):
    os.makedirs(folder)
folder = folder + "/"

def plot_loss(ech,loss,ech_a,acc):
    plt.plot(ech,loss)
    plt.xlabel("num of epoch")
    plt.ylabel("loss")
    plt.title("Loss Against Epoch")
    plt.savefig("loss.png")
    ##################################
    plt.clf()
    plt.plot(ech_a,acc)
    plt.xlabel("num of epoch")
    plt.ylabel("accuracy")
    plt.title(f"accuracy Against Epoch {np.max(acc)}%")
    plt.savefig(folder+"acc.png")
    ##################################
def plot_dist(dataset,output_size,title=""):
    plt.clf()
    x = range(output_size)
    count = bf.view_count(dataset,output_size)
    with open(f"{folder}{title}_distrubtion.txt","w+") as f:
        for i in range(len(count)):
            f.write(f"{i},{count[i]}\n")
    plt.bar(x,count)
    plt.xlabel("space group num within Hex range")
    plt.ylabel("num of data")
    plt.title(f"Total_size = {len(dataset)}")
    plt.savefig(f"{folder}{title}_distrubtion.png")

def main_one(csnum):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # prepare neural network
    validate_size = 0.1
    num_bands = 100
    tmp = []
    for hs in range(48):
        tmp.append(hs)
    hs_indices = tmp
    #hs_indices = [0, 1, 3, 4, 5, 7, 8, 13, 31, 34, 37]  # 11 hs points in Brillouin zone out of 40

    cs_sizes = crystalsystem.crystalsystem_sizes()
    output_size = cs_sizes[csnum - 1] - cs_sizes[csnum - 2] + 1 if csnum > 1 else 3
    """
    model = torch.nn.Sequential(
        torch.nn.LeakyReLU(),
        torch.nn.Linear(len(hs_indices)*num_bands, 300),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(300, 100),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(100, output_size),
        torch.nn.LeakyReLU(),
    )
    """
    model = torch.nn.Sequential(
        #torch.nn.LeakyReLU(),
        torch.nn.Linear(len(hs_indices)*num_bands, 128),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, output_size),
        #torch.nn.LeakyReLU(),
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)
    criterion = torch.nn.CrossEntropyLoss()

    with open ("data.pickle","rb") as f:
        dataset = pickle.load(f)

    # data cut off and shuffle
    data_in,data_out = bf.data_cutoff(dataset,output_size,cut_off=70)
    data_loader.update_dataset(dataset,data_in,data_out)
    # spilt dataset
    #train_dataset,test_dataset = data_loader.spilt_train_test_dataset(dataset)
    train_dataset,test_dataset = data_loader.advanced_spilt_train_test_dataset(dataset,output_size)
     balance train part
    train_in,train_out = bf.balance_avg(train_dataset,output_size)
    print(f"Before balance:\n{bf.view_count(train_dataset,output_size)}")
    data_loader.update_dataset(train_dataset,train_in,train_out)
    print(f"After balance:\n{bf.view_count(train_dataset,output_size)}")
    validate_loader = data_loader.get_validate_loader(test_dataset,32)
    train_loader = data_loader.get_train_loader(train_dataset,32)

    # train
    ech,loss,ech_a,acc = function_training.validate_train_loop(
        device, model, optimizer, scheduler, criterion, validate_loader, train_loader,
        num_epoch=30, num_epoch_per_validate=1, state_dict_path=f"state_dicts/state_dict_cs2sg_{csnum}"
    )

    plot_loss(ech,loss,ech_a,acc)
    plot_dist(dataset,output_size,title="Cut-off Raw sample")
    plot_dist(train_dataset,output_size,title="Train sample")
    plot_dist(test_dataset,output_size,title="Test sample")
    """
    # apply
    function_list.append_any_guess_list_files(
        device, model, hs_indices, validate_size, num_group=230,
        in_list_paths=[f"list/actual/spacegroup_list_{sgnum}.txt" for sgnum in crystalsystem.spacegroup_number_range(csnum)],
        out_list_path_format="list/guess/spacegroup_list_{}.txt"
    )
    """
if __name__ == '__main__':
    function_list.create_empty_list_files(230, out_list_path_format="list/guess/spacegroup_list_{}.txt")
    """
    for i in range(1, 8):
        main_one(i)
    """
    main_one(6)
