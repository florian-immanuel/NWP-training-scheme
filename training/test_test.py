import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print ("MPS device not found.")

    # Parameters that can and should be changed
    path_to_training_data = "../training_sets/daily_dialog/100/8-1_16-1_32-1_64-1_128-1/"

    log_file = f"{path_to_training_data.lstrip('./').lstrip('training_sets/').rstrip('/').rstrip('1234567890-_')}"

    print(log_file)