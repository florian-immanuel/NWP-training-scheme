import torch
import os
import time
import random
import cpuinfo
from torch.utils.data import DataLoader, TensorDataset
from transformers import MobileBertTokenizer, MobileBertForMaskedLM
from preprocessing.blend_training_data import blend_split_data_epoch_wise
from evaluation.acc_bert_func import bert_accuracy


#give the training scheme a name that reflects the training scheme
#this will be in the name_of_training_schema of the model and the logfile
name_of_training_schema = "1xalldata_10epochs_all_equal"
#Parameters that can and should be changed
max_length = 128
model_name = 'google/mobilebert-uncased'
num_epochs = 10
batch_number = 100
batch_size=32
settings = ((20,20,20,20,20),(20,20,20,20,20),(20,20,20,20,20),(20,20,20,20,20),(20,20,20,20,20),
            (20,20,20,20,20),(20,20,20,20,20),(20,20,20,20,20),(20,20,20,20,20),(20,20,20,20,20))

def train(name_of_training_schema = name_of_training_schema, max_length = max_length, model_name = model_name,
          num_epochs = num_epochs, batch_number = batch_number, batch_size=batch_size,
          settings = settings):

#Pre Settings:
    time0=time.time()
    path_to_training_data = blend_split_data_epoch_wise(shuffeled_datasets="_shuffled", dataset="daily_dialog", epochs=10, splits=1000, dataset_type="train",
                                                        name = f"{name_of_training_schema}",
                                                        settings=settings)

    path_to_log = f"../logs/{path_to_training_data.lstrip('./').lstrip('training_sets/').rstrip('/').rstrip('qwertzuiopasdfghjklyxcvbnm1234567890-_')}"
    log_file = f"../logs/{path_to_training_data.lstrip('./').lstrip('training_sets/').rstrip('/')}.txt"
    if not os.path.exists(path_to_log):
        os.makedirs(path_to_log)



    ####    Parameters that can and should NOT be changed   ###
    tokenizer = MobileBertTokenizer.from_pretrained(model_name)
    model = MobileBertForMaskedLM.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device("mps") #this would run on apple GPU which is slower for small models
    model.to(device)  #move model to device (CPU or GPU)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    if device == torch.device("mps"):
        training_on = "GPU"
    else:
        training_on = "CPU"

    ####    Initiate Log file   ###
    with open(f'{log_file}', 'w') as loss_log:
        loss_log.write(f"Model: mobileBERT \n Settings: {settings} \n Training on: {training_on}")

    print(f"Training on {training_on}")

    ###     Initiate String for Log     ###
    logs = ""

    ###     Log title       ###
    logs += f"{name_of_training_schema}\n\n"
    ###     log name and settings of GPU or CPU    ###
    if torch.cuda.is_available():
        print(f"CUDA verfügbar: {torch.cuda.is_available()}")
        print(f"Anzahl der verfügbaren GPUs: {torch.cuda.device_count()}")
        logs += f"Anzahl der verfügbaren GPUs: {torch.cuda.device_count()}"
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logs += f"GPU {i}: {torch.cuda.get_device_name(i)}"
    else:
        info = cpuinfo.get_cpu_info()

        for key, value in info.items():
            print(f"{key}: {value}")
            logs+=f"{key}: {value}"
        print("\n ###BEGINNING TRAINING###\n")
        logs += "\n ###BEGINNING TRAINING###\n"


    global total_training_instances
    total_training_instances = 0

    ###     Begin of TRAINING   ###
    for epoch in range(num_epochs):
        time1 = time.time()
        logs = ""
        print(f"Start epoch {epoch+1}")
        logs += f"\n \t #######################   EPOCH {epoch + 1}   ##############################\n \n"

    ###     Open Training files     ###
        with open(f'{path_to_training_data}{epoch}.txt', 'r') as f:
            lines = f.readlines()
    ########Shuffle whole dataset#############
            random.shuffle(lines)

    ###     Number of training files per Epoch and in total     ###
        training_instances = len(lines)
        # total_training_instances
        total_training_instances += training_instances

        print(f"opening {training_instances} files")
        logs += f"This Epoch has {training_instances} training instances\n"

    ###     Define input and target files   ###
        inputs = [(input_text + " [MASK] .") for input_text in (line.split(',')[0] for line in lines)]
        targets = [(input_text.replace(",", " ").replace('\n',' .')) for input_text in (line for line in lines)]


    ###     Log and measure time    ###

        print("training files completed")
        time2=time.time()
        print(f"in time: {time2-time1} \n")

        #Define batch size
        batch_size = training_instances//batch_number
        logs += f"Training instances per batch: {batch_size} \n Number of batches: {batch_number}\n\n"

        #Define loop for each batch
        for batch in range(batch_number):
            #Measure time and Log
            time3=time.time()
            print(f"Training on batch {batch+1} of {batch_number}")
            print(f"Training instances: {batch_size}")
            #logs += f"Batch: {batch + 1} / {batch_number}\t"

            #Move through training examples
            start_index = batch*batch_size
            end_index = start_index+batch_size



            #########Define and encode input and target
            input_encodings = tokenizer(inputs[start_index:end_index], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            target_encodings = tokenizer(targets[start_index:end_index], return_tensors="pt", padding=True, truncation=True, max_length=max_length)

            if input_encodings['input_ids'].size(0) != target_encodings['input_ids'].size(0):
                print(
                    f"Warning: Mismatched tokenized lengths. Input tokenized length: {input_encodings['input_ids'].size(0)}, Target tokenized length: {target_encodings['input_ids'].size(0)}. Skipping this batch.")
                continue


    #########Test Training data ########### The shuffle of training data works fine :)
    #        print(inputs[start_index:end_index][-5:])
     #       print("\n")
      #      print(targets[start_index:end_index][-5:])


            #Load the Dataloader
            dataset = TensorDataset(input_encodings['input_ids'], input_encodings['attention_mask'],
                                    target_encodings['input_ids'])
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            #Train the model
            model.train()
            total_loss = 0


            for input_ids, attention_mask, target_ids in dataloader:
                optimizer.zero_grad()

                #move data to GPU if possible
                input_ids, attention_mask, target_ids = [x.to(device) for x in [input_ids, attention_mask, target_ids]]
                if input_ids.size() ==target_ids.size():
                    outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)

                    loss = outputs.loss
                    total_loss += loss.item()

                    loss.backward()
                    optimizer.step()

            time4 = time.time()
            print(f"Total time for batch: {time4 - time3}")
            print(f"Loss: {total_loss/batch_size}")

    ###     Evaluate every 25 batches     ###
            if (batch + 1) % 25 == 0:  # Using 10 as the modulus since it's the LCM of 5 and 10

                    print("Validating")
                    performance = f"{bert_accuracy(input_model=model, input_tokenizer=tokenizer, percentage=0.02, test_dataset=False)}\n"
                    logs += f"Epoch: {epoch}  Batch: {batch+1}: {performance}"
                    print(performance)
                    output_dir = f'../models/{name_of_training_schema}Epoch_{epoch}_{batch + 1}/'
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)



            with open(f'{log_file}', 'a') as loss_log:
                loss_log.write(logs)
            logs = ""
            print(f"Batch {batch+1} of {batch_number} completed\n")

    ###     Test performance   at end of epoch  ###
        performance = f"\n\n{bert_accuracy(input_model=model, input_tokenizer=tokenizer, percentage=0.1, test_dataset=False)}\n"
        logs += f"Epoch: {epoch} \n {performance}"

        ###     write all logs of this epoch to file    ###
        with open(f'{log_file}', 'a') as loss_log:
            loss_log.write(logs)
        print(performance)
        logs = ""

    ###     safe model    ###
        output_dir = f'../models/{name_of_training_schema}Epoch_{epoch}_{batch + 1}_3/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"Epoch {epoch+1} completed\n")
