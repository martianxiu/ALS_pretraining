import re
from os.path import join
import argparse
import pickle
import os

def get_laz_file_paths(directory):
    laz_file_paths = []
    
    # Walk through the directory and subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".laz"):
                laz_file_paths.append(os.path.join(root, file))
    
    return laz_file_paths

def create_info(split, tile_paths, root):
    # create the dictionary 
    info_list = []
    num_features = len(["x", "y", "z", "intensity", "return_number", "num_return", "return_ratio"])
    region_name = split
    for tp in tile_paths:
        # print(tp)
        index_start = tp.split('/').index(split)
        tile_idx = '/'.join(tp.split('/')[index_start+1:])
        # print(tile_idx)
        
        point_cloud = {
            'num_features': num_features,
            'region_name': region_name,
            'sample_idx': tile_idx
        }
        sample_dict = {"point_cloud": point_cloud}
        info_list.append(sample_dict)
        # print(info_list)
        # break
        
    dump_path = join(root, f"{region_name}.pkl")
    dump_pickle_file(info_list, dump_path)
    print(f"{dump_path} dumped.")
    # break
    

def dump_pickle_file(data, file_path):
    """
    Save an object to a pickle file at the specified file path.

    Args:
    data (object): The object to be pickled.
    file_path (str): The path to the pickle file.

    Returns:
    None
    """
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def extract_path(input_string):
    """
    Extracts the file path from a given string.

    Parameters:
    input_string (str): The input string containing the file path.

    Returns:
    str: The extracted file path.
    """
    # Use regular expression to match the path inside the parentheses
    match = re.search(r"'(.*?)'", input_string)
    if match:
        return match.group(1)
    else:
        return None


parser = argparse.ArgumentParser(description='generate info')
parser.add_argument('-i', '--input_dir', type=str, required=True, help='The path to the input dir.')

args = parser.parse_args()

input_dir = args.input_dir

root_training = join(input_dir, "train")
training_tiles = get_laz_file_paths(root_training)

root_test = join(input_dir, "test")
test_tiles = get_laz_file_paths(root_test)

root_val = join(input_dir, "val")
val_tiles = get_laz_file_paths(root_val)

create_info("train", training_tiles, root_training)
create_info("test", test_tiles, root_test)
create_info("val", val_tiles, root_val)

