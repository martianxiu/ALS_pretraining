# from datasets import load_dataset

# dataset = load_dataset("IGNF/PureForest", local_dir=".")
# print(dataset.cache_files)

from huggingface_hub import snapshot_download

# Define the repository id and the directory where you want to download the files
repo_id = "IGNF/PureForest"  # Example: Replace this with the dataset you need
local_dir = "../"  # Example: Specify your desired local directory

# Download the dataset to the specified local directory
snapshot_download(repo_id, repo_type="dataset", local_dir=local_dir)