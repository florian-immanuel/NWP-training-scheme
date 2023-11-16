import csv
import os
from collections import Counter


def inspect_results(input_file_name, inspect_only_wrongs= False, save_as_file=False, output_file_name=None):
    """
    Inspects the results from the input file and returns or saves the results based on the given options.
    """
    right=[]  # List of correct predictions
    wrong=[]  # List of wrong predictions

    with open(input_file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[2] in row[1]:  # If the target word is in the generated words
                right.append((row[2], row[1]))
            else:
                wrong.append((row[2], row[1]))

    if save_as_file == False:
        # Return both wrong and right predictions
        if inspect_only_wrongs == False:
            return f"right: {right}",f"\n"*20, f"wrong: {wrong}"
        # Only return wrong predictions
        if inspect_only_wrongs == True:
            return wrong

    if save_as_file == True:

        # Check if directory exists, if not create one
        if not os.path.exists(f"../evaluation/inspect_results/"):
            os.makedirs(f"../evaluation/inspect_results")

        file_name_wrong = f"{output_file_name}_wrong.txt"
        file_name_right = f"{output_file_name}_right.txt"

        # Save the wrong predictions to a file
        with open(f"../evaluation/inspect_results/{file_name_wrong}", mode="w", newline="") as save_file:
            writer = csv.writer(save_file)
            for element in wrong:
                writer.writerow(element)
            print(f"Elements saved to {file_name_wrong}")

        # Save the correct predictions to a file
        with open(f"../evaluation/inspect_results/{file_name_right}", mode="w", newline="") as save_file:
            writer = csv.writer(save_file)
            for element in wrong:
                writer.writerow(element)
            print(f"Elements saved to {file_name_right}")

    return f"results have been saved"


# Get only the wrong predictions
only_wrong = inspect_results("../evaluation/daily_dialog/test_3-10_words_BERT.txt", inspect_only_wrongs=True)

# Extract just the words from the wrong predictions
wrong_words = [word[0] for word in only_wrong]

# Uncomment the following line to check the count of each wrong word
#print(Counter(wrong_words)

# List of result files
results = ["../evaluation/daily_dialog/test_4-8_words_BERT.txt", "../evaluation/daily_dialog/test_4-16_words_BERT.txt","../evaluation/daily_dialog/test_4-32_words_BERT.txt",
           "../evaluation/daily_dialog/test_4-64_words_BERT.txt","../evaluation/daily_dialog/test_4-128_words_BERT.txt","../evaluation/daily_dialog/test_4-256_words_BERT.txt"]


