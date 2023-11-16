import pandas as pd

def process_txt_to_csv(input_txt_path: str, output_csv_path: str) -> None:
    """
    Process the contents of a .txt file (in the format of validate_4-16_words.txt) and save it as a .csv file.

    Parameters:
    - input_txt_path (str): Path to the input .txt file.
    - output_csv_path (str): Path where the output .csv file will be saved.

    Returns:
    None
    """

    data = {
        "input": [],
        "target": []
    }

    with open(input_txt_path, "r") as file:
        for line in file:
            input_text, target_word = line.strip().split(',')
            input_text += ' [MASK]'  # Add mask token at the end of each sentence
            data["input"].append(input_text)
            data["target"].append(target_word)

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)

    # Add an index column similar to test.csv
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Unnamed: 0"}, inplace=True)

    # Save the DataFrame to the specified output CSV file
    df.to_csv(output_csv_path, index=False)

# Test the function with the provided file
paths = [("../data/daily_dialog/validate/validate_4-8_words.txt","../data/daily_dialog/validate/xvalidate-8_words.csv"),
                ("../data/daily_dialog/validate/validate_4-16_words.txt","../data/daily_dialog/validate/xvalidate-16_words.csv"),
                ("../data/daily_dialog/validate/validate_4-32_words.txt","../data/daily_dialog/validate/xvalidate-32_words.csv"),
                ("../data/daily_dialog/validate/validate_4-64_words.txt","../data/daily_dialog/validate/xvalidate-64_words.csv"),
                ("../data/daily_dialog/validate/validate_4-128_words.txt","../data/daily_dialog/validate/xvalidate-128_words.csv")]
for path in paths:
    process_txt_to_csv(path[0], path[1])
