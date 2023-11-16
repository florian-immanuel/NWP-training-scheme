from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import pandas as pd
import torch.nn.functional as F



maximum_lengths = [8,16,32,64, 128]
#maximum_lengths = [1,5,10,25, 50]
datasets = [8, 16, 32, 64, 128]
#for dataset in datasets:
for max_len in maximum_lengths:
#for dataset in datasets:

    #models
    #model_name = f'mBERT_Conversation_epoch_{max_len}'
    model_name = f'tBERT_Daily_Dialog_max_len_{max_len}epoch_1'
    #model_name = "TinyBERT"

    model_path = f'../../../models/{model_name}'
    #model_path = f"huawei-noah/TinyBERT_General_4L_312D"
    #model_path = f"google/mobilebert-uncased"

    # Load model and tokenizer
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load test data
    #test_data = pd.read_csv(f"../../../data/daily_dialog/test/xtest_4-{dataset}_words.csv").sample(frac=0.02)
    test_data = pd.read_csv(f"../../../data/Conversation/test.csv")

# Initialize lists to store log-likelihoods with counts
    log_likelihoods = {"2_tokens": [], "2_to_6_tokens": [], "7_to_11_tokens": [], "12_or_more_tokens": [], "overall": []}
    counts = {"2_tokens": 0, "2_to_6_tokens": 0, "7_to_11_tokens": 0, "12_or_more_tokens": 0, "overall": 0}

    # Iterate over  test data
    for index, row in test_data.iterrows():
        # Tokenize  input and target
        tokens = tokenizer(row['input'], return_tensors='pt', add_special_tokens=False)
        target_token_id = tokenizer.convert_tokens_to_ids(row['target'])

        # Get predictions
        predictions = model(**tokens)
        predicted_probs = F.softmax(predictions.logits[0, -1, :], dim=-1)
        target_prob = predicted_probs[target_token_id].item()

        # Compute log-likelihood of target word
        log_likelihood = torch.log(torch.tensor(target_prob)).item()

        # Append result to appropriate list based on input length
        num_tokens = len(tokens['input_ids'][0])
        if num_tokens == 2:
            log_likelihoods["2_tokens"].append(log_likelihood) # 1 context token
            counts["2_tokens"] += 1
        elif 2 < num_tokens <= 6:
            log_likelihoods["2_to_6_tokens"].append(log_likelihood) # 2 to 5 context tokens
            counts["2_to_6_tokens"] += 1
        elif 6 < num_tokens <= 11:
            log_likelihoods["7_to_11_tokens"].append(log_likelihood) # 6 to 10 context tokens
            counts["7_to_11_tokens"] += 1
        else:
            log_likelihoods["12_or_more_tokens"].append(log_likelihood) # more than 10 context tokens
            counts["12_or_more_tokens"] += 1

        # Append result to overall list
        log_likelihoods["overall"].append(log_likelihood)
        counts["overall"] += 1

    # Calculate average log-likelihood for each category and overall

    #with open(f'../../../evaluation/log_likelihood/matrix/{model_name}_dataset_{dataset}.txt', 'w') as f:
    with open(f'../../../evaluation/log_likelihood/matrix/{model_name}_Conversation.txt', 'w') as f:
        for category, result in log_likelihoods.items():
            if result:  # Check if result is not empty
                avg_log_likelihood = sum(result) / len(result)
                print(f"Average log-likelihood for {category}: {avg_log_likelihood}")
                print(f"Number of sentences in {category}: {counts[category]}")
                f.write(f"Average log-likelihood for {category}: {avg_log_likelihood}\n")
                f.write(f"Number of sentences in {category}: {counts[category]}\n")
            else:
                print(f"No sentences of {category} length in test data.")
                f.write(f"No sentences of {category} length in test data.\n")