import os
import laspy
import numpy as np

import os
import laspy
import numpy as np

def split_las_file(input_file, output_dir, window_size):
    # Load the .las file
    las = laspy.read(input_file)
    
    # Extract point coordinates
    x_coords = np.array(las.x)
    y_coords = np.array(las.y)

    # sliding window 
    max_x = np.max(x_coords)
    max_y = np.max(y_coords)
    min_x = np.min(x_coords)
    min_y = np.min(y_coords)

    num_window_x = ((max_x - min_x) // window_size) + 1
    num_window_y = ((max_y - min_y) // window_size) + 1

    x_start = min_x
    y_start = min_y

    for i in range(int(num_window_x)):
        
        if i == (num_window_x - 1): 
            # last part will take from the boundary
            x_end = max_x
            x_start = max_x - window_size
        else: 
            # otherwise slide
            x_end = x_start + window_size

        for j in range(int(num_window_y)):
            if j == (num_window_y - 1):
                y_end = max_y
                y_start = max_y - window_size
            else:
                y_end = y_start + window_size

            # crop. 
            mask = np.logical_and(
                np.logical_and(x_coords >= x_start, x_coords <= x_end),
                np.logical_and(y_coords >= y_start, y_coords <= y_end)
            )

            tile_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
            tile_las.points = las.points[mask]
            
            # Save the tile to the output directory
            output_file = os.path.join(output_dir, f"{os.path.basename(input_file).replace('.las', '')}_tile_x{i}_y{j}.las")
            tile_las.write(output_file)
            print(f"Saved {output_file}")
            
            # update the y 
            y_start = y_end

        # update the x and reset the y
        x_start = x_end
        y_start = min_y

def split_all_las_files(input_dir, output_dir, window_size):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each .las file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".las"):
            input_file = os.path.join(input_dir, filename)
            split_las_file(input_file, output_dir, window_size)
            print(f"Processed {input_file}")

if __name__ == "__main__":
    # Specify input and output directories
    input_directory = "../test"
    output_directory = "../val"

    split_all_las_files(input_directory, output_directory, window_size=144)
