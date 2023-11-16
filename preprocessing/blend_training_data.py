import os
import random
def blend_split_data(shuffeled_datasets="_shuffled", dataset="daily_dialog", epochs=10, splits=100, dataset_type="train", w8=20, w16=20, w32=10, w64=10, w128=10):
    """blends the training data; important: shuffle all has to be false if ascend wants to be tested
    it returns a string to the training data, number if epoch is still missing"""
    a, b, c, d, e = 0, 0, 0, 0, 0


    for epoch in range(epochs):
        datasets_list = []
        # Use separate variables for the loop iteration (i, j, k, l, m)
        for i in range(w8):
            datasets_list.append(
                f"../data/{dataset}/split_datasets{shuffeled_datasets}/{splits}/{dataset_type}/train_4-8_words/train_4-8_words_{(a + i) % (splits - 1)}.txt")
        for j in range(w16):
            datasets_list.append(
                f"../data/{dataset}/split_datasets{shuffeled_datasets}/{splits}/{dataset_type}/train_4-16_words/train_4-16_words_{(b + j) % (splits - 1)}.txt")
        for k in range(w32):
            datasets_list.append(
                f"../data/{dataset}/split_datasets{shuffeled_datasets}/{splits}/{dataset_type}/train_4-32_words/train_4-32_words_{(c + k) % (splits - 1)}.txt")
        for l in range(w64):
            datasets_list.append(
                f"../data/{dataset}/split_datasets{shuffeled_datasets}/{splits}/{dataset_type}/train_4-64_words/train_4-64_words_{(d + l) % (splits - 1)}.txt")
        for m in range(w128):
            datasets_list.append(
                f"../data/{dataset}/split_datasets{shuffeled_datasets}/{splits}/{dataset_type}/train_4-128_words/train_4-128_words_{(e + m) % (splits - 1)}.txt")

        a += w8
        b += w16
        c += w32
        d += w64
        e += w128

        if not os.path.exists(f"../training_sets/{dataset}/{splits}/8-{w8}_16-{w16}_32-{w32}_64-{w64}_128-{w128}/"):
            os.makedirs(f"../training_sets/{dataset}/{splits}/8-{w8}_16-{w16}_32-{w32}_64-{w64}_128-{w128}/")
        with open(f"../training_sets/{dataset}/{splits}/8-{w8}_16-{w16}_32-{w32}_64-{w64}_128-{w128}/{epoch}.txt", "w") as out_file:

            ##only shuffles the parts of the dataset and not each line
            random.shuffle(datasets_list)
            for data_path in datasets_list:
                with open(data_path, "r") as infile:
                    lines = infile.readlines()
                    ##shuffles all lines
                    random.shuffle(lines)
                    out_file.writelines(lines)


    return f"../training_sets/{dataset}/{splits}/8-{w8}_16-{w16}_32-{w32}_64-{w64}_128-{w128}/"



def blend_split_data_epoch_wise(shuffeled_datasets="_shuffled", dataset="daily_dialog", epochs=10, splits=100, dataset_type="train",
                                name = "training_pattern",
                                settings=((20,20,10,10,10),(20,20,10,10,10),(20,20,10,10,10),(20,20,10,10,10),(20,20,10,10,10),
                                        (20,20,10,10,10),(20,20,10,10,10),(20,20,10,10,10),(20,20,10,10,10),(20,20,10,10,10))):
    if len(settings) != epochs:
        print("Settings must be same number as epochs")


    """blends the training data; important: every epoch has to be given
    it returns a string to the training data, number if epoch is still missing"""
    a, b, c, d, e = 0, 0, 0, 0, 0


    for epoch in range(epochs):
        w8, w16, w32, w64, w128 = settings[epoch][0], settings[epoch][1], settings[epoch][2], settings[epoch][3], settings[epoch][4]
        datasets_list = []
        # Use separate variables for the loop iteration (i, j, k, l, m)
        for i in range(w8):
            datasets_list.append(
                f"../data/{dataset}/split_datasets{shuffeled_datasets}/{splits}/{dataset_type}/train_4-8_words/train_4-8_words_{(a + i) % (splits - 1)}.txt")
        for j in range(w16):
            datasets_list.append(
                f"../data/{dataset}/split_datasets{shuffeled_datasets}/{splits}/{dataset_type}/train_4-16_words/train_4-16_words_{(b + j) % (splits - 1)}.txt")
        for k in range(w32):
            datasets_list.append(
                f"../data/{dataset}/split_datasets{shuffeled_datasets}/{splits}/{dataset_type}/train_4-32_words/train_4-32_words_{(c + k) % (splits - 1)}.txt")
        for l in range(w64):
            datasets_list.append(
                f"../data/{dataset}/split_datasets{shuffeled_datasets}/{splits}/{dataset_type}/train_4-64_words/train_4-64_words_{(d + l) % (splits - 1)}.txt")
        for m in range(w128):
            datasets_list.append(
                f"../data/{dataset}/split_datasets{shuffeled_datasets}/{splits}/{dataset_type}/train_4-128_words/train_4-128_words_{(e + m) % (splits - 1)}.txt")

        a += w8
        b += w16
        c += w32
        d += w64
        e += w128

        if not os.path.exists(f"../training_sets/{dataset}/{splits}/{name}/"):
            os.makedirs(f"../training_sets/{dataset}/{splits}/{name}/")
        with open(f"../training_sets/{dataset}/{splits}/{name}/{epoch}.txt", "w") as out_file:

            ##only shuffles the parts of the dataset and not each line
            random.shuffle(datasets_list)
            for data_path in datasets_list:
                with open(data_path, "r") as infile:
                    lines = infile.readlines()
                    ##shuffles all lines
                    random.shuffle(lines)
                    out_file.writelines(lines)


    return f"../training_sets/{dataset}/{splits}/{name}/"

