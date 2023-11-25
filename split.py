import os

def split_file(file_path, output_dir):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return
    
    # Check if the output directory exists, create it if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the size of the file
    file_size = os.path.getsize(file_path)

    # Calculate the midpoint to split the file into two equal parts
    midpoint = file_size // 2

    # Open the input file for reading
    with open(file_path, 'rb') as input_file:
        # Read the first half of the file
        first_half = input_file.read(midpoint)

        # Create the output paths for the two parts
        first_half_path = os.path.join(output_dir, 'part1_' + os.path.basename(file_path))
        second_half_path = os.path.join(output_dir, 'part2_' + os.path.basename(file_path))

        # Write the first half to the first output file
        with open(first_half_path, 'wb') as output_file:
            output_file.write(first_half)

        # Read the second half of the file
        second_half = input_file.read()

        # Write the second half to the second output file
        with open(second_half_path, 'wb') as output_file:
            output_file.write(second_half)

    print(f"File successfully split into two equal parts: {first_half_path} and {second_half_path}")

# Example usage
input_file_path = r'/home/ajs667/federated_pix2pix_real/pytorch-CycleGAN-and-pix2pix/datasets/example_split/PreEtch_9wafer_TGAP_ZiwangPairs/AB_1'
output_directory = r'/home/ajs667/federated_pix2pix_real/pytorch-CycleGAN-and-pix2pix/datasets/example_split/PreEtch_9wafer_TGAP_ZiwangPairs'
split_file(input_file_path, output_directory)
