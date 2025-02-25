'''
    generate train_val_test split by combinng Training and Validation_raw tiles 
'''

import os
import numpy as np


def get_all_laz_files(root_dir):
    """
    Get a list of all .laz files in the given root directory, including subdirectories.

    Args:
        root_dir (str): Path to the root directory.

    Returns:
        list: List of paths to .laz files.
    """
    laz_files = []
    
    # Walk through the root directory and its subdirectories
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".laz"):
                laz_files.append(os.path.join(dirpath, filename))
    
    return laz_files

def write_list_to_file(data_list, file_path):
    """
    Writes a list to a text file, with each element on a new line.

    Args:
        data_list (list): The list to write to the file.
        file_path (str): The path to the output text file.

    Returns:
        None
    """
    try:
        with open(file_path, 'w') as file:
            for item in data_list:
                file.write(f"{item}\n")
        print(f"List successfully written to {file_path}")
    except Exception as e:
        print(f"An error occurred while writing the file: {e}")

# Example usage
if __name__ == "__main__":

    np.random.seed(41)

    train_laz = get_all_laz_files('Training')
    val_laz = get_all_laz_files('Validation_raw')
    
    tile_bank = {}
    for laz in train_laz:
        # major_class = laz.split('/')[1]
        minor_class = laz.split('/')[2]
        tile_name = laz.split('/')[3]
        if tile_bank.get(minor_class, None) is not None:
            existing_val = tile_bank[minor_class]
            new_val = existing_val + [laz]
            tile_bank[minor_class] = new_val
        else:
            tile_bank[minor_class] = [laz]

    # directly assign val files to the test. as they are carefully chosen. see OpenGF paper
    test_files = val_laz
    
    # assign  more tiles per major scene class to test files
    num_pop = 2
    for key in list(tile_bank.keys()):
        pop_items = []
        val = tile_bank[key]
        for _ in range(num_pop):
            pop_items.append(val.pop(np.random.randint(0, len(val) - 1)))
        test_files += pop_items
    print(test_files)
    print(len(test_files))

    val_files = []
    # assign  more tiles per major scene class to val files
    num_pop = 3
    for key in list(tile_bank.keys()):
        pop_items = []
        val = tile_bank[key]
        for _ in range(num_pop):
            pop_items.append(val.pop(np.random.randint(0, len(val) - 1)))
        val_files += pop_items
    print(val_files)
    print(len(val_files))

    # flatten all remaining files in the tile bacnk
    train_files = []
    for key in list(tile_bank.keys()):
        val = tile_bank[key]
        train_files += val
    print(train_files)
    print(len(train_files))

    total = len(train_files) + len(val_files) + len(test_files)
    print(f'Total: {total}. train: {len(train_files) / total * 100}%, val: {len(val_files) / total * 100}%, test: {len(test_files) / total * 100}%')


    # write to text files 
    write_list_to_file(train_files, 'train.txt')
    write_list_to_file(val_files, 'val.txt')
    write_list_to_file(test_files, 'test.txt')
    

