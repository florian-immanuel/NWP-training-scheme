import torch
import os
import time
import random
from torch.utils.data import DataLoader, TensorDataset
from transformers import MobileBertTokenizer, MobileBertForMaskedLM
from preprocessing.blend_training_data import blend_split_data
from evaluation.acc_bert_func import bert_accuracy


settings = [("shuffled_datasets=","_shuffled"),("dataset=","daily_dialog"), ("epochs=",10), ("splits=",1000), ("dataset_type=","train"),
            ("w8",14), ("w16",20), ("w32",10), ("w64", 6), ("w128",0)]


#Pre Settings:
path_to_training_data = blend_split_data(settings[0][1],settings[1][1],settings[2][1],settings[3][1],settings[4][1],
                                         settings[5][1],settings[6][1],settings[7][1],settings[8][1],settings[9][1])
path_to_log = f"../logs/{path_to_training_data.lstrip('./').lstrip('training_sets/').rstrip('/').rstrip('1234567890-_')}"
log_file = f"../logs/{path_to_training_data.lstrip('./').lstrip('training_sets/').rstrip('/')}.txt"
if not os.path.exists(path_to_log):
    os.makedirs(path_to_log)


#Parameters that can and should be changed
max_length = 128
model_name = 'google/mobilebert-uncased'
model_name = "/Users/Immanuel/PycharmProjects/NWP-training-schema/models/train69"
num_epochs = 10
batch_number = 100


#Parameters that can and should NOT be changed
tokenizer = MobileBertTokenizer.from_pretrained(model_name)
model = MobileBertForMaskedLM.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device("mps") #this would run on apple GPU which is slower for small models
model.to(device)  #move model to device (CPU or GPU)
optimizer =  torch.optim.AdamW(model.parameters(), lr=2e-5)

if device == torch.device("mps"):
    training_on = "GPU"
else:
    training_on = "CPU"

# Initiate Log file
with open(f'{log_file}', 'w') as loss_log:
    loss_log.write(f"Model: mobileBERT \n Settings: {settings} \n Training on: {training_on}")

print(f"Training on {training_on}")

#Initiate String for Log
logs = ""

for epoch in range(num_epochs):
    time0 = time.time()
    print(f"Start epoch {epoch+1}")
    logs += f"\n \t Start epoch {epoch + 1}\n \n"

    # Open Training files
    with open(f'{path_to_training_data}{epoch}.txt', 'r') as f:
        lines = f.readlines()
########Shuffle whole dataset
        random.shuffle(lines)

    #Number of training files
    training_instances = len(lines)
    print(f"opening {training_instances} files")

#####Define input and target files
    inputs = [(input_text + " [MASK] .") for input_text in (line.split(',')[0] for line in lines)]
    targets = [(input_text.replace(",", " ").replace('\n',' .')) for input_text in (line for line in lines)]



    #Log and measure time
    logs += f"Total number of training instances: {training_instances}\n"
    time1=time.time()
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
        logs += f"Batch: {batch + 1} / {batch_number}\t"

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
#########Test Training data
#        print(inputs[start_index:end_index][-5:])
 #       print("\n")
  #      print(targets[start_index:end_index][-5:])


        #Load the Dataloader
        dataset = TensorDataset(input_encodings['input_ids'], input_encodings['attention_mask'],
                                target_encodings['input_ids'])
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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
        logs += f"Loss: {total_loss / batch_size}\n Time:{round(time4 - time3, 2)}  "

#########Evaluate and safe every 5 or 20 batches
        match (batch + 1) % 10:  # Using 10 as the modulus since it's the LCM of 5 and 10
            case 5:  # Equivalent to (batch+1) % 5 == 0
                print("Validating")
                performance = f"{bert_accuracy(input_model=model, input_tokenizer=tokenizer, percentage=0.005, test_dataset=False)}\n"
                logs += performance
                print(performance)
            case 0:  # Equivalent to (batch+1) % 10 == 0
                print("Validating")
                performance = f"{bert_accuracy(input_model=model, input_tokenizer=tokenizer, percentage=0.01, test_dataset=False)}\n"
                logs += performance
                print(performance)
                output_dir = f'../models/train{epoch}_{batch+1}/'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
            case _:
                pass




        with open(f'{log_file}', 'a') as loss_log:
            loss_log.write(logs)
        logs = ""
        print(f"Batch {batch+1} of {batch_number} completed\n")


    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(dataloader)}")

