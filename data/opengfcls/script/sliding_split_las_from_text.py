import os
import laspy
import numpy as np



def split_las_file(input_file, output_dir, window_size, stride, min_points):
    """
    Splits a LAS/LAZ file into patches using a sliding window with a defined stride.
    Discards patches with fewer points than the specified threshold.

    Args:
        input_file (str): Path to the input LAS/LAZ file.
        output_dir (str): Path to the directory to save the output patches.
        window_size (float): Size of the sliding window.
        stride (float): Step size for the sliding window.
        min_points (int): Minimum number of points required to save a patch.

    Returns:
        None
    """
    # Load the .las file
    las = laspy.read(input_file)
    
    # Extract point coordinates
    x_coords = np.array(las.x)
    y_coords = np.array(las.y)

    # Sliding window boundaries
    max_x = np.max(x_coords)
    max_y = np.max(y_coords)
    min_x = np.min(x_coords)
    min_y = np.min(y_coords)

    # Initialize the starting points for the sliding window
    x_start = min_x

    while x_start < max_x:
        x_end = x_start + window_size
        if x_end > max_x:  # Ensure the last patch covers the boundary
            x_end = max_x

        y_start = min_y
        while y_start < max_y:
            y_end = y_start + window_size
            if y_end > max_y:  # Ensure the last patch covers the boundary
                y_end = max_y

            # Create a mask to filter points within the current window
            mask = np.logical_and(
                np.logical_and(x_coords >= x_start, x_coords <= x_end),
                np.logical_and(y_coords >= y_start, y_coords <= y_end)
            )

            # Check the number of points in the patch
            if np.sum(mask) >= min_points:
                # Create a new LAS object for the current patch
                tile_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
                tile_las.points = las.points[mask]

                # Preserve the header metadata
                tile_las.header.scales = las.header.scales
                tile_las.header.offsets = las.header.offsets
                
                # Save the tile to the output directory
                filename = input_file.split('/')[-1] # xxx.laz
                output_file = os.path.join(
                    output_dir, 
                    f"{os.path.basename(filename).replace('.laz', '')}_tile_x{int(x_start)}_y{int(y_start)}.laz"
                )
                tile_las.write(output_file)
                print(f"Saved {output_file}")
            else:
                print(f"Skipped patch at x={x_start}, y={y_start} due to insufficient points.")

            # Move the sliding window in the Y direction
            y_start += stride

        # Move the sliding window in the X direction
        x_start += stride

def read_text_file(file_path):
    """
    Reads a text file and returns its content as a string.

    Args:
        file_path (str): Path to the text file.

    Returns:
        str: The content of the text file.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: For other I/O errors.
    """
    try:
        with open(file_path, 'r') as file:
            content = file.readlines()
            content = [x.strip() for x in content]
        return content
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        raise
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        raise


def split_all_las_files_from_list(file_paths, output_dir, window_size, stride, min_points):
    """
    Splits a list of LAS/LAZ files into smaller patches using a sliding window.

    Args:
        file_paths (list): List of full file paths to LAS/LAZ files.
        output_dir (str): Directory where output patches will be saved.
        window_size (int): Size of the sliding window.

    Returns:
        None
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for input_file in file_paths:
        # Check if the file has the correct extension
        if input_file.endswith(".las") or input_file.endswith(".laz"):
            # Process the LAS/LAZ file
            split_las_file(input_file, output_dir, window_size, stride, min_points)
            print(f"Processed {input_file}")

if __name__ == "__main__":
    # input_text_name = 'train'
    # input_file_list = read_text_file(os.path.join('script', 'train_val_test_split', input_text_name + '.txt'))
    
    # window_size = 100 # meter
    # stride_size = 50 # meter
    # min_points = 2000
    # output_directory = f"{input_text_name}_cls_w{window_size}_s{stride_size}"

    # split_all_las_files_from_list(input_file_list, output_directory, window_size=window_size, stride=stride_size, min_points=min_points)


    input_text_name = 'test'
    input_file_list = read_text_file(os.path.join('script', 'train_val_test_split', input_text_name + '.txt'))
    
    window_size = 100 # meter
    stride_size = 100 # meter
    min_points = 2000
    output_directory = f"{input_text_name}_cls_w{window_size}_s{stride_size}"

    split_all_las_files_from_list(input_file_list, output_directory, window_size=window_size, stride=stride_size, min_points=min_points)

    input_text_name = 'val'
    input_file_list = read_text_file(os.path.join('script', 'train_val_test_split', input_text_name + '.txt'))
    
    window_size = 100 # meter
    stride_size = 100 # meter
    min_points = 2000
    output_directory = f"{input_text_name}_cls_w{window_size}_s{stride_size}"

    split_all_las_files_from_list(input_file_list, output_directory, window_size=window_size, stride=stride_size, min_points=min_points)
