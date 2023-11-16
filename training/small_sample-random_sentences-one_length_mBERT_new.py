import torch
import os
import time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW
from torch.utils.data import DataLoader, TensorDataset
from evaluation.acc_bert_func import bert_accuracy

"""Here we will train on small_sample_size, \n
random_sentences from the dataset \n
one_length during each epoch \n
Model is mBERT \n
"""
train_settings = "small_sample_shuffled_sentences_one_length"

# Initialize base model
model_name = 'google/mobilebert-uncased'

tokenizer = AutoTokenizer.from_pretrained(f'{model_name}')
model = AutoModelForMaskedLM.from_pretrained(f"{model_name}")

# Define device
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=1e-5, no_deprecation_warning=True)

def add_mask(text):
    if not isinstance(text, str):
        raise ValueError(f"Expected a string, but received {type(text)} with value {text}")

    tokens = tokenizer.tokenize(text)
    if tokens:
        if tokens[-1] != "[MASK]":
            tokens.append(tokenizer.mask_token)
    return tokenizer.convert_tokens_to_string(tokens)

# numbers of epochs

num_epochs = 10

chunk_size= num_rows = 10000


with open(f'../logs/train-{train_settings}_{num_epochs}.txt', 'w') as loss_log:
    loss_log.write(f"Model: mobileBERT \n Settings: {train_settings} \n")

    for epoch in range(num_epochs):
        num_sentences = 0
        # Create an iterator for reading data in chunks for the new epoch
        #percent_and_max_len = ((0.0003, 18), (0.0004, 10), (0.0003, 20), (0.0003, 40), (0.01, 70))
        percent_and_max_len = ((0.1, 10), (0.1, 20), (0.1, 40), (0.1, 70), (0.1, 140))

        if epoch in[0,5]:
            training_data_0 = pd.read_csv(f"../data/daily_dialog/train/train_4-8_words.txt")
            training_data_0_csv = training_data_0.to_csv(f"temporary_training_data/dd_{percent_and_max_len[0][0]}_{percent_and_max_len[0][1]}")
            setting = percent_and_max_len[0]

        if epoch in [1, 6]:
            training_data_1 = pd.read_csv(f"../data/daily_dialog/train/train_4-16_words.txt")
            training_data_1_csv = training_data_1.to_csv(f"temporary_training_data/dd_{percent_and_max_len[1][0]}_{percent_and_max_len[1][1]}")
            setting = percent_and_max_len[1]

        if epoch in [2, 7]:
            training_data_2 = pd.read_csv(f"../data/daily_dialog/train/train_4-32_words.txt")
            training_data_2_csv = training_data_2.to_csv(f"temporary_training_data/dd_{percent_and_max_len[2][0]}_{percent_and_max_len[2][1]}")
            setting = percent_and_max_len[2]

        if epoch in [3, 8]:
            training_data_3 = pd.read_csv(f"../data/daily_dialog/train/train_4-64_words.txt")
            training_data_3_csv = training_data_3.to_csv(f"temporary_training_data/dd_{percent_and_max_len[3][0]}_{percent_and_max_len[3][1]}")
            setting = percent_and_max_len[3]

        if epoch in [4, 9]:
            training_data_4 = pd.read_csv(f"../data/daily_dialog/train/train_4-128_words.txt")
            training_data_4_csv = training_data_4.to_csv(f"temporary_training_data/dd_{percent_and_max_len[4][0]}_{percent_and_max_len[4][1]}")
            setting = percent_and_max_len[4]


        print(f'Starting epoch number {epoch + 1}_{setting[1]}...')

        data_iterator = pd.read_csv(f"temporary_training_data/dd_{setting[0]}_{setting[1]}", names=["index", "input", "target"], chunksize=chunk_size, skiprows=1)

#how m


        epoch_loss = 0.0


        for chunk_num, chunk in enumerate(data_iterator):
            print(f"Epoch: {epoch+1}, {setting[0]} of Dataset: {setting[1]}, Data chunk {chunk_num + 1}")

            # Data preprocessing and DataLoader creation code
            df = chunk.sample(frac=0.5)

            df = df.dropna(subset=['input', 'target'])
            df['input'] = df['input'].apply(lambda x: str(x))
            df['target'] = df['target'].apply(lambda x: str(x))

            df.columns = ["index", "input", "target"]

            try:
                inputs = [tokenizer.encode_plus(add_mask(text), add_special_tokens=False, truncation=True,
                        max_length=setting[1], padding='max_length') for text in df["input"] if isinstance(text, str)]
            except ValueError as e:
                print(f"Error processing text data in inputs: {e}")

            targets = [tokenizer.encode(text, add_special_tokens=False)[0] for text in df['target'] ]
            input_ids = torch.tensor([item['input_ids'] for item in inputs])
            attention_masks = torch.tensor([item['attention_mask'] for item in inputs])
            target_ids = torch.tensor(targets)

            # Create DataLoader
            dataset = TensorDataset(input_ids, attention_masks, target_ids)
            dataloader = DataLoader(dataset, batch_size=32)

            # Training code
            for batch in dataloader:
                # Unpack the batch and move tensors to GPU if available
                b_input_ids, b_attention_masks, b_target_ids = [b.to(device) for b in batch]
                # Create label tensor
                labels = -100 * torch.ones_like(b_input_ids)
                # Get MASK token pos
                mask_positions = (b_input_ids == tokenizer.mask_token_id).nonzero(as_tuple=False)
                # Replace -100 at mask pos
                for i, pos in enumerate(mask_positions):
                    labels[pos[0], pos[1]] = b_target_ids[i]

                optimizer.zero_grad()

                # Forward pass
                outputs = model(b_input_ids, attention_mask=b_attention_masks, labels=labels)

                # Compute loss
                loss = outputs.loss
                epoch_loss += loss.item()

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

            # Reset the iterator for the next epoch
            data_iterator = pd.read_csv(f"temporary_training_data/dd_{setting[0]}_{setting[1]}", header=None,
                                        delimiter=',', chunksize=chunk_size)

        print(f'Epoch {epoch + 1} loss: {epoch_loss / (10 * len(dataloader))}')
        loss_log.write(f"\n Epoch {epoch + 1} \t Time:{time.ctime()} \t Sentences: {num_sentences} \t"
        f"accuracy on dataset {i}: {bert_accuracy(input_model=model, input_tokenizer=tokenizer)} \t"
        f" Loss: {epoch_loss / (10 * len(dataloader))}")


        #print(f"accuracy  {i}: {bert_accuracy(input_model=model, input_tokenizer=tokenizer)} \n")
        #print(f"Time in Hours: {time.ctime()}")

            # Saving model
        output_dir = f'../models/train{percent_and_max_len}{epoch + 1}/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    #loss_log.write(f"accuracy on test: {bert_accuracy(input_model=model, input_tokenizer=tokenizer)} \n")
    #print(f"accuracy on test: {bert_accuracy(input_model=model, input_tokenizer=tokenizer)} \n")

