import numpy as np
import pickle
import torch.nn.functional
import balance_function as bf
import data_loader
import function_training
import function_list

import crystalsystem
import matplotlib.pyplot as plt


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
        #torch.nn.LeakyReLU(),
        torch.nn.Linear(len(hs_indices)*num_bands, 64),
        #torch.nn.LeakyReLU(),
        #torch.nn.Linear(64, 32),

        torch.nn.Conv1d(64, [32], 1),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(32, output_size),
        #torch.nn.LeakyReLU(),
        #torch.nn.Softmax(dim=7),
    )
    """
    model = torch.nn.Sequential(
        torch.nn.LeakyReLU(),
        torch.nn.Linear(len(hs_indices)*num_bands, 128),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(128, 32),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(32, output_size),
        torch.nn.LeakyReLU(),
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)
    criterion = torch.nn.CrossEntropyLoss()

    # prepare data
    crystal_upper = crystalsystem.spacegroup_index_upper(csnum)
    crystal_lower = crystalsystem.spacegroup_index_lower(csnum)
    crystal_size = crystal_upper - crystal_lower


    def json2inputlabel(data_json):
        data_input_np = np.array(data_json["bands"])[:, hs_indices].flatten().T
        sgnum = data_json["number"]
        if crystal_lower < sgnum - 1 < crystal_upper:
            data_label_np = np.array([sgnum - 1 - crystal_lower])
        else:
            data_label_np = np.array([crystal_size])
        return data_input_np, data_label_np

    dataset = data_loader.AnyDataset(
        [f"list/actual/spacegroup_list_{sgnum}.txt" for sgnum in crystalsystem.spacegroup_number_range(csnum)],
        json2inputlabel, validate_size
    )

    with open ("data.pickle","wb+") as f:
        pickle.dump(dataset,f)
    

    with open ("data.pickle","rb") as f:
        dataset = pickle.load(f)

    train_data,train_label = bf.balance(dataset,output_size)
    dataset.data_inputs = train_data
    dataset.data_labels = train_label
    dataset.update_inform()


    """
    print(len(dataset))
    y = np.zeros(27)
    for i in range(len(dataset)):
        index = int(dataset[i][1])
        y[index] += 1

    x = range(len(y))
    plt.bar(x,y)
    plt.title(f"Total_size = {len(dataset)}")
    plt.xlabel("space group num within Hex range")
    plt.ylabel("num of data")
    plt.savefig("report/original_sample_set")
    exit()
    """


    validate_loader, train_loader = data_loader.get_validate_train_loader(dataset, 32)

    # train
    ech,loss,ech_a,acc = function_training.validate_train_loop(
        device, model, optimizer, scheduler, criterion, validate_loader, train_loader,
        num_epoch=5, num_epoch_per_validate=1, state_dict_path=f"state_dicts/state_dict_cs2sg_{csnum}"
    )

    plt.plot(ech,loss)
    plt.savefig("loss.png")
    plt.clf()
    plt.plot(ech_a,acc)
    plt.savefig("acc.png")

    # apply
    function_list.append_any_guess_list_files(
        device, model, hs_indices, validate_size, num_group=230,
        in_list_paths=[f"list/actual/spacegroup_list_{sgnum}.txt" for sgnum in crystalsystem.spacegroup_number_range(csnum)],
        out_list_path_format="list/guess/spacegroup_list_{}.txt"
    )

if __name__ == '__main__':
    function_list.create_empty_list_files(230, out_list_path_format="list/guess/spacegroup_list_{}.txt")
    main_one(6)
    exit()
    for i in range(1, 8):
        main_one(i)

    #main_one(6)
