import numpy as np
import pickle
import torch.nn.functional

import data_loader
import function_training
import function_list

import crystalsystem
import function_analysis
import matplotlib.pyplot as plt


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # prepare neural network
    validate_size = 0.1
    num_bands = 100
    """
    tmp = []
    for hs in range(48):
        tmp.append(hs)
    hs_indices = tmp
    """
    hs_indices = [0]
    #hs_indices = range(48)
    """
    model = torch.nn.Sequential(
        #torch.nn.LeakyReLU(),
        torch.nn.Linear(len(hs_indices)*num_bands, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(32, 7),
        #torch.nn.LeakyReLU(),
        #torch.nn.Softmax(dim=7),
    )
    """
    # https://pytorch.org/docs/stable/nn.html


    model = torch.nn.Sequential(
        torch.nn.LeakyReLU(),
        torch.nn.Linear(len(hs_indices)*num_bands, 128),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(128, 32),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(32, 7),
        torch.nn.LeakyReLU(),
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)
    criterion = torch.nn.CrossEntropyLoss()

    # prepare data
    def json2inputlabel(data_json):
        data_input_np = np.array(data_json["bands"])[:, hs_indices].flatten().T
        data_label_np = np.array([crystalsystem.crystalsystem_number(data_json["number"]) - 1])
        return data_input_np, data_label_np
    """
    dataset = data_loader.AnyDataset(
        [f"list/actual/crystalsystem_list_{csnum}.txt" for csnum in range(1, 8)],
        json2inputlabel, validate_size
    )

    #validate_loader, train_loader = data_loader.get_validate_train_loader(dataset, 32)
    with open ("data.pickle","wb+") as f:
        pickle.dump(dataset,f)

    """
    with open ("data.pickle","rb") as f:
        dataset = pickle.load(f)
    validate_loader, train_loader = data_loader.get_validate_train_loader(dataset, 20)

    # train
    ech,loss,ech_a,acc = function_training.validate_train_loop(
        device, model, optimizer, scheduler, criterion, validate_loader, train_loader,
        num_epoch=20, num_epoch_per_validate=3, state_dict_path="state_dicts/state_dict_bs2cs",load_data=True
    )
    plt.plot(ech,loss)
    plt.savefig("loss.png")
    plt.clf()
    plt.plot(ech_a,acc)
    plt.savefig("acc.png")
    # apply
    function_list.create_any_guess_list_files(
        device, model, hs_indices, validate_size, num_group=7,
        in_list_paths=[f"list/actual/crystalsystem_list_{csnum}.txt" for csnum in range(1, 8)],
        out_list_path_format="list/guess/crystalsystem_list_{}.txt"
    )


    # analyse
    function_analysis.print_result(
        group_numbers=range(1, 8),
        guess_list_dir="list/guess/",
        actual_list_dir="list/actual/",
        list_format="crystalsystem_list_{}.txt",
        validate_size=0.1
    )

    def json2label(data_json):
        data_label_np = np.array([crystalsystem.crystalsystem_number(data_json["number"]) - 1])
        return data_label_np
    function_analysis.show_confusion(
        json2label,
        [f"list/guess/crystalsystem_list_{i}.txt" for i in range(1, 8)],
        group_names=["TRI", "MCL", "ORC", "TET", "TRG", "HEX", "CUB"],
        title="bs2cs"
    )


if __name__ == "__main__":
    main()
