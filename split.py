import os
import shutil
import random

def split_directory(input_dir, output_dir1, output_dir2):
    # Check if the input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Directory '{input_dir}' not found.")
        return
    
    # Check if the output directories exist, create them if not
    for output_dir in [output_dir1, output_dir2]:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Get the list of image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    # Shuffle the list of image files
    random.shuffle(image_files)

    # Calculate the midpoint to split the list into two equal parts
    midpoint = len(image_files) // 2

    # Split the list into two equal parts
    first_half = image_files[:midpoint]
    second_half = image_files[midpoint:]

    # Move files from the input directory to the first output directory
    for file_name in first_half:
        source_path = os.path.join(input_dir, file_name)
        destination_path = os.path.join(output_dir1, file_name)
        shutil.move(source_path, destination_path)

    # Move files from the input directory to the second output directory
    for file_name in second_half:
        source_path = os.path.join(input_dir, file_name)
        destination_path = os.path.join(output_dir2, file_name)
        shutil.move(source_path, destination_path)

    print(f"Directory successfully split into two equal parts: {output_dir1} and {output_dir2}")

# Example usage
input_directory = '/home/ajs667/federated_pix2pix_real/pytorch-CycleGAN-and-pix2pix/datasets/example_split/PreEtch_9wafer_TGAP_ZiwangPairs/AB_1/train'
output_directory1 = '/home/ajs667/federated_pix2pix_real/pytorch-CycleGAN-and-pix2pix/datasets/example_split/part1'
output_directory2 = '/home/ajs667/federated_pix2pix_real/pytorch-CycleGAN-and-pix2pix/datasets/example_split/part2'
split_directory(input_directory, output_directory1, output_directory2)
