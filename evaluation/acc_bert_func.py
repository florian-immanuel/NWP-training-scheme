from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import pandas as pd
import torch.nn.functional as F


#maximum_lengths = [8,16,32,64, 128]
def bert_accuracy(input_model, input_tokenizer, percentage=0.01, test_dataset=False):

    datasets = [8, 16, 32, 64, 128]

    accuracy_list = []
    model = input_model  # AutoModelForMaskedLM.from_pretrained(model_path)
    tokenizer = input_tokenizer  # AutoTokenizer.from_pretrained(model_path)

    for dataset in datasets:


            # Load test data
            if test_dataset:
                test_data = pd.read_csv(f"../data/daily_dialog/test"
                                    f"/xtest_4-{dataset}_words.csv").sample(frac=percentage)
            else:
                test_data = pd.read_csv(f"../data/daily_dialog/validate"
                                    f"/xvalidate-{dataset}_words.csv").sample(frac=percentage)

            # Initialize lists to store accuracy with counts
            correct_counts = {"2_tokens": 0, "2_to_6_tokens": 0, "7_to_11_tokens": 0, "12_or_more_tokens": 0,
                              "overall": 0}
            counts = {"2_tokens": 0, "2_to_6_tokens": 0, "7_to_11_tokens": 0, "12_or_more_tokens": 0, "overall": 0}

            # Iterate over  test data
            for index, row in test_data.iterrows():
                # Tokenize  input and target
                tokens = tokenizer(row['input'], return_tensors='pt', add_special_tokens=False)
                target_token_id = tokenizer.convert_tokens_to_ids(row['target'])

                # Get predictions
                predictions = model(**tokens)
                predicted_probs = F.softmax(predictions.logits[0, -1, :], dim=-1)

                # Check if target is in top 3 predictions
                top_predicted_token_ids = torch.topk(predicted_probs, 3).indices.tolist()
                is_correct = target_token_id in top_predicted_token_ids

                num_tokens = len(tokens['input_ids'][0])

                # Increment only correct counts
                if is_correct:
                    correct_counts["overall"] += 1

                    if num_tokens == 2:
                        correct_counts["2_tokens"] += 1
                        counts["2_tokens"] += 1
                    elif num_tokens <= 6:
                        correct_counts["2_to_6_tokens"] += 1
                        counts["2_to_6_tokens"] += 1
                    elif num_tokens <= 11:
                        correct_counts["7_to_11_tokens"] += 1
                        counts["7_to_11_tokens"] += 1
                    else:
                        correct_counts["12_or_more_tokens"] += 1
                        counts["12_or_more_tokens"] += 1

                        # Increment count based on input length
                counts["overall"] += 1

                if num_tokens == 2:
                    counts["2_tokens"] += 1
                elif num_tokens <= 6:
                    counts["2_to_6_tokens"] += 1
                elif num_tokens <= 11:
                    counts["7_to_11_tokens"] += 1
                else:
                    counts["12_or_more_tokens"] += 1


                # Calculate accuracy
            for category in ["overall"]:
                if counts[category] > 0:
                    accuracy = correct_counts[category] / counts[category]
                    accuracy_list.append(f"validate {dataset} = {round(accuracy, 3)}")
                    print(f"Accuracy for {dataset} {category}: {round(accuracy, 3)}")

    return accuracy_list
