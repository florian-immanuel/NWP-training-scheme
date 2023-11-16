import csv
import os
from datasets import load_dataset
import random
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW


# Load the daily_dialog dataset from the HuggingFace's datasets library
list_datasets = ["daily_dialog", "persona_chat"]
dataset_name = "daily_dialog"
dataset = load_dataset(f"{dataset_name}")

# specific to daily dialog dataset
train_dataset = dataset["train"].remove_columns(["emotion", "act"])["dialog"]
validate_dataset = dataset["validation"].remove_columns(["emotion", "act"])["dialog"]
test_dataset = dataset["test"].remove_columns(["emotion", "act"])["dialog"]


# Convert the dataset to lower case, remove commas, and make each conversation a single string
train_dataset_uncased_one_string = [[" ".join(word.lower().replace(",", "") for word in inner_list)]
    for inner_list in train_dataset]
validate_dataset_uncased_one_string = [[" ".join(word.lower().replace(",", "") for word in inner_list)]
    for inner_list in validate_dataset]
test_dataset_uncased_one_string = [[" ".join(word.lower().replace(",", "") for word in inner_list)]
    for inner_list in test_dataset]

# Define a function to re-format the dataset to predict the next word in a sentence
def train_every_next_word(input_list, min_words=3, max_words=30):
    list_every_next_word = []
    for conversation in input_list:
        counter = min_words
        conversation_string = conversation[0].split()
        for index in range(len(conversation_string)):
            if counter == len(conversation_string)- min_words:
                break
            if min_words <= counter < len(conversation_string):
                if counter < max_words:
                    list_every_next_word.append([" ".join(conversation_string[:counter]), conversation_string[counter]])
                else:
                    sentence_cut_off = counter - max_words
                    list_every_next_word.append([" ".join(conversation_string[sentence_cut_off:counter]), conversation_string[counter]])
            counter += 1
    return list_every_next_word


def remove_multiple_token_words_as_target(input_list=None, model_name='google/mobilebert-uncased'):
    """Removes all lines that have a target word made up of more than one token
    It overwrites the file it is given"""

    tokenizer = AutoTokenizer.from_pretrained(f'{model_name}')
    list_one_token_targetwords = []
    counter = 0

    for  line in input_list:
        #print(line[-1])
        counter += 1
        target_word = line[-1]
        if len(tokenizer.tokenize(target_word)) == 1 and str(target_word) not in [".", "!", "?", "'", "’"]:
            list_one_token_targetwords.append(line)
            #print(line)
    print(f"Created list with words. Before: {counter} After: {len(list_one_token_targetwords)}")

    return list_one_token_targetwords

# Function to save list in file or display
def save_list_to_csv(input_list, name, directory, original=False):
    if not os.path.exists(f"../data/{dataset_name}/{directory}/"):
        os.makedirs(f"../data/{dataset_name}/{directory}/")

    file_path = f"../data/{dataset_name}/{directory}/{name}.txt"

    # Write the content of input_list to the file
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        for element in input_list:
            writer.writerow(element[:])
            file.flush()
            #split each file in number_of_splits

    print(f"{len(input_list)} Lines saved to {file_path}")

def split_list_to_csv_shuffle(input_list, name, directory, number_of_splits=100):
    if not os.path.exists(f"../data/{dataset_name}/split_datasets_shuffled/{number_of_splits}/{directory}/{name}/"):
        os.makedirs(f"../data/{dataset_name}/split_datasets_shuffled/{number_of_splits}/{directory}/{name}/")

    divider = len(input_list)//number_of_splits
    random.shuffle(input_list)
    # Write the content of input_list to the file
    for number in range(number_of_splits):
        split_file_path = f"../data/{dataset_name}/split_datasets_shuffled/{number_of_splits}/{directory}/{name}/{name}_{number}.txt"  # muss in loop

        start_index = divider*number
        end_index = divider*(number+1)

        with open(split_file_path, mode="w", newline="") as file:
            writer = csv.writer(file)

            writer.writerows(input_list[start_index:end_index])
            file.flush()

    print(f"{number_of_splits} split_files saved to {split_file_path}")


def split_list_to_csv(input_list, name, directory, number_of_splits=100):
    if not os.path.exists(f"../data/{dataset_name}/split_datasets/{number_of_splits}/{directory}/{name}/"):
        os.makedirs(f"../data/{dataset_name}/split_datasets/{number_of_splits}/{directory}/{name}/")

    divider = len(input_list)//number_of_splits
    # Write the content of input_list to the file
    for number in range(number_of_splits):
        split_file_path = f"../data/{dataset_name}/split_datasets/{number_of_splits}/{directory}/{name}/{name}_{number}.txt"  # muss in loop

        start_index = divider*number
        end_index = divider*(number+1)

        with open(split_file_path, mode="w", newline="") as file:
            writer = csv.writer(file)

            writer.writerows(input_list[start_index:end_index])
            file.flush()

    print(f"{number_of_splits} split_files saved to {split_file_path}")



if __name__ == "__main__":

    # Define the datasets for processing in a loop
    train = train_dataset_uncased_one_string
    validate = validate_dataset_uncased_one_string
    test = test_dataset_uncased_one_string
    all_datasets = [(train, "train"), (validate, "validate"), (test, "test")]

    # Define different sequence lengths for processing
    different_sequence_lengths = [(4, 8), (4, 16), (4, 32), (4, 64), (4, 128), (4, 256)]
    number_of_splits = 1000

    # Process each dataset and save to file for different sequence lengths
    for dataset in all_datasets:
        current_dataset = dataset[0]
        #save_list_to_csv(current_dataset, f"{dataset[1]}_original", f"{dataset[1]}", original=True)
        for i in range(len(different_sequence_lengths)):
            output = remove_multiple_token_words_as_target(input_list=train_every_next_word(current_dataset, different_sequence_lengths[i][0], different_sequence_lengths[i][1]))
            save_list_to_csv(output, f"{dataset[1]}_{different_sequence_lengths[i][0]}-{different_sequence_lengths[i][1]}_words", f"{dataset[1]}")
            split_list_to_csv_shuffle(output, f"{dataset[1]}_{different_sequence_lengths[i][0]}-{different_sequence_lengths[i][1]}_words", f"{dataset[1]}", number_of_splits=number_of_splits)
            split_list_to_csv(output, f"{dataset[1]}_{different_sequence_lengths[i][0]}-{different_sequence_lengths[i][1]}_words", f"{dataset[1]}", number_of_splits=number_of_splits)

    print("DONE")